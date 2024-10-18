from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

import polars as pl
from bcc import BPF, PerfType
from data_schema import CollectionTable, TLBPerfTable

from data_collection.bpf_instrumentation.bpf_hook import BPFProgram


class PerfHWCacheConfig:
  # From perf_hw_cache_id in uapi/linux/perf_event.h
  class Cache(Enum):
    PERF_COUNT_HW_CACHE_L1D = 0
    PERF_COUNT_HW_CACHE_L1I = 1
    PERF_COUNT_HW_CACHE_LL = 2
    PERF_COUNT_HW_CACHE_DTLB = 3
    PERF_COUNT_HW_CACHE_ITLB = 4
    PERF_COUNT_HW_CACHE_BPU = 5
    PERF_COUNT_HW_CACHE_NODE = 6

  # From perf_hw_cache_op_id in uapi/linux/perf_event.h
  class Op(Enum):
    PERF_COUNT_HW_CACHE_OP_READ = 0
    PERF_COUNT_HW_CACHE_OP_WRITE = 1
    PERF_COUNT_HW_CACHE_OP_PREFETCH = 2

  # From perf_hw_cache_op_result_id  in uapi/linux/perf_event.h
  class Result(Enum):
    PERF_COUNT_HW_CACHE_RESULT_ACCESS = 0
    PERF_COUNT_HW_CACHE_RESULT_MISS = 1

  # From https://man7.org/linux/man-pages/man2/perf_event_open.2.html
  @classmethod
  def config(cls, cache: Cache, op: Op, result: Result) -> int:
    return (cache.value) | (op.value << 8) | (result.value << 16)


PERF_HANDLER: Final[str] = """
BPF_PERF_OUTPUT(NAME);
int NAME_on(struct bpf_perf_event_data* ctx) {
  struct bpf_perf_event_value value_buf;
  if (bpf_perf_prog_read_value(ctx, (void*)&value_buf, sizeof(struct bpf_perf_event_value))) {
    return 0;
  }
  struct perf_event_data data;
  __builtin_memset(&data, 0, sizeof(data));
  u64 ts = bpf_ktime_get_ns();
  data.ts_uptime_us = ts / 1000;
  data.count = value_buf.counter;
  data.enabled_time_us = value_buf.enabled / 1000;
  data.running_time_us = value_buf.running / 1000;
  NAME.perf_submit(ctx, &data, sizeof(data));
  return 0;
}
"""

@dataclass(frozen=True)
class PerfData:
  cpu: int
  ts_uptime_us: int
  cumulative_tlb_misses: int
  pmu_enabled_time_us: int
  pmu_running_time_us: int

  @classmethod
  def from_event(cls, cpu: int, event: Any):
    return PerfData(
        cpu=cpu,
        ts_uptime_us=event.ts_uptime_us,
        cumulative_tlb_misses=event.count,
        pmu_enabled_time_us=event.enabled_time_us,
        pmu_running_time_us=event.running_time_us,
    )


class TLBPerfBPFHook(BPFProgram):

  @classmethod
  def name(cls) -> str:
    return "tlb_perf"

  def __init__(self):
    self._perf_data = dict[str, list[PerfData]]()
    bpf_text = open(Path(__file__).parent / "bpf/tlb_perf.bpf.c", "r").read()
    # add perf handlers
    universal_perf_events = ["dtlb_misses", "itlb_misses"]
    for perf_event in universal_perf_events:
      bpf_text += PERF_HANDLER.replace("NAME", perf_event)
      self._perf_data[perf_event] = list[PerfData]()
    self.bpf_text = bpf_text

  def load(self, collection_id: str):
    self.collection_id = collection_id
    self.bpf = BPF(text = self.bpf_text)
    # sample frequency is in hertz
    self.bpf.attach_perf_event(
      ev_type=PerfType.HW_CACHE,
      ev_config=PerfHWCacheConfig.config(
        cache=PerfHWCacheConfig.Cache.PERF_COUNT_HW_CACHE_DTLB,
        op=PerfHWCacheConfig.Op.PERF_COUNT_HW_CACHE_OP_READ,
        result=PerfHWCacheConfig.Result.PERF_COUNT_HW_CACHE_RESULT_MISS,
      ),
      fn_name=b"dtlb_misses_on",
      sample_freq=1000,
    )
    self.bpf.attach_perf_event(
      ev_type=PerfType.HW_CACHE,
      ev_config=PerfHWCacheConfig.config(
        cache=PerfHWCacheConfig.Cache.PERF_COUNT_HW_CACHE_ITLB,
        op=PerfHWCacheConfig.Op.PERF_COUNT_HW_CACHE_OP_READ,
        result=PerfHWCacheConfig.Result.PERF_COUNT_HW_CACHE_RESULT_MISS,
      ),
      fn_name=b"itlb_misses_on",
      sample_freq=1000,
    )
    for event_name in self._perf_data.keys():
      self.bpf[event_name].open_perf_buffer(self._perf_handler(event_name), page_cnt=64)

  def poll(self):
    self.bpf.perf_buffer_poll()

  def close(self):
    self.bpf.cleanup()

  def data(self) -> list[CollectionTable]:
    dtlb_df = pl.DataFrame(self._perf_data["dtlb_misses"]).with_columns(
      pl.lit(True).alias("dtlb_event"), pl.lit(False).alias("itlb_event")
    )
    itlb_df = pl.DataFrame(self._perf_data["itlb_misses"]).with_columns(
      pl.lit(False).alias("dtlb_event"), pl.lit(True).alias("itlb_event")
    )
    return [
      TLBPerfTable.from_df_id(
        pl.concat([dtlb_df, itlb_df]),
        collection_id=self.collection_id,
      ),
    ]

  def clear(self):
    self._perf_data.clear()

  def pop_data(self) -> list[CollectionTable]:
    tlb_miss_tables = self.data()
    self.clear()
    return tlb_miss_tables

  def _perf_handler(self, event_name: str):
    def _perf_event_handler(cpu, perf_event_data, size):
      event = self.bpf[event_name].event(perf_event_data)
      try:
          self._perf_data[event_name].append(PerfData.from_event(cpu, event))
      except Exception as _:
        pass
    return _perf_event_handler
