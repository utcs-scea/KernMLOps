from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import plotext as plt
import polars as pl
from bcc import BPF

from data_collection.bpf_instrumentation.bpf_hook import BPFProgram


@dataclass(frozen=True)
class QuantaRuntimeData:
  cpu: int
  pid: int
  tgid: int
  quanta_end_uptime_us: int
  quanta_run_length_us: int

class QuantaRuntimeBPFHook(BPFProgram):

  @classmethod
  def name(cls) -> str:
    return "quanta_runtime"

  def __init__(self):
    self.is_support_raw_tp = BPF.support_raw_tracepoint()
    bpf_text = open(Path(__file__).parent / "bpf/sched_quanta_runtime.bpf.c", "r").read()

    # code substitutions
    if BPF.kernel_struct_has_field(b'task_struct', b'__state') == 1:
        bpf_text = bpf_text.replace('STATE_FIELD', '__state')
    else:
        bpf_text = bpf_text.replace('STATE_FIELD', 'state')
    # pid from userspace point of view is thread group from kernel pov
    # bpf_text = bpf_text.replace('FILTER', 'tgid != %s' % args.pid)
    self.bpf_text = bpf_text.replace('FILTER', '0')
    if self.is_support_raw_tp:
        bpf_text = bpf_text.replace('USE_TRACEPOINT', '1')
    else:
        bpf_text = bpf_text.replace('USE_TRACEPOINT', '0')
    self.quanta_runtime_data = list[QuantaRuntimeData]()

  def load(self):
    self.bpf = BPF(text = self.bpf_text)
    if not self.is_support_raw_tp:
      self.bpf.attach_kprobe(
        event_re=rb'^finish_task_switch$|^finish_task_switch\.isra\.\d$',
        fn_name=b"trace_run"
      )
    self.bpf["quanta_runtimes"].open_perf_buffer(self._event_handler)

  def poll(self):
    self.bpf.perf_buffer_poll()

  def data(self) -> pl.DataFrame:
    return pl.DataFrame(self.quanta_runtime_data)

  def clear(self):
    self.quanta_runtime_data.clear()

  def pop_data(self) -> pl.DataFrame:
    quanta_df = self.data()
    self.clear()
    return quanta_df

  def _event_handler(self, cpu, quanta_runtime_perf_event, size):
    event = self.bpf["quanta_runtimes"].event(quanta_runtime_perf_event)
    self.quanta_runtime_data.append(
      QuantaRuntimeData(
        cpu=cpu,
        pid=event.pid,
        tgid=event.tgid,
        quanta_end_uptime_us=event.quanta_end_uptime_us,
        quanta_run_length_us=event.quanta_run_length_us,
      )
    )

  @classmethod
  def plot(cls,
    collections_dfs: Mapping[str, pl.DataFrame],
    collection_id: str | None = None
  ) -> None:
    if cls.name() not in collections_dfs:
      return

    # TODO(Patrick): Make this function tolerate multiple collections
    quanta_df = collections_dfs[cls.name()]
    system_info_df = collections_dfs["system_info"]
    if collection_id:
      quanta_df = quanta_df.filter(
        pl.col("collection_id") == collection_id
      )
      system_info_df = system_info_df.filter(
        pl.col("collection_id") == collection_id
      )
    benchmark_start_time_sec = system_info_df.select("uptime_sec").to_series()[0]

    # filter out invalid data points due to data loss
    quanta_df = quanta_df.filter(
      pl.col("quanta_run_length_us") < 5_000
    )

    # group by and plot by cpu
    quanta_df_by_cpu = quanta_df.group_by("cpu")
    for cpu, quanta_df_group in quanta_df_by_cpu:
      plt.scatter(
        (
          (quanta_df_group.select("quanta_end_uptime_us") / 1_000_000) - benchmark_start_time_sec
        ).to_series().to_list(),
        quanta_df_group.select("quanta_run_length_us").to_series().to_list(),
        label=f"CPU {cpu[0]}",
      )
    plt.title("Quanta Runtimes")
    plt.xlabel("Benchmark Runtime (usec)")
    plt.ylabel("Quanta Run Length (usec)")
    plt.show()
    graph_dir = Path(f"data/graphs/{collection_id}")
    graph_dir.mkdir(parents=True, exist_ok=True)
    plt.save_fig(str(graph_dir / f"{cls.name()}.plt"), keep_colors=True)
