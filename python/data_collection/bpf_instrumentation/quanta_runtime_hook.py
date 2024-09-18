from dataclasses import dataclass
from pathlib import Path

import polars as pl
from bcc import BPF

from bpf_instrumentation.bpf_hook import BPFProgram


@dataclass(frozen=True)
class QuantaRuntimeData:
  cpu: int
  pid: int
  quanta_end_uptime_us: int
  quanta_run_length_us: int

class QuantaRuntimeBPFHook(BPFProgram):

  def __init__(self):
    bpf_text = open(Path(__file__).parent / "bpf/sched_quanta_runtime.bpf.c", "r").read()

    # code substitutions
    if BPF.kernel_struct_has_field(b'task_struct', b'__state') == 1:
        bpf_text = bpf_text.replace('STATE_FIELD', '__state')
    else:
        bpf_text = bpf_text.replace('STATE_FIELD', 'state')
    # pid from userspace point of view is thread group from kernel pov
    # bpf_text = bpf_text.replace('FILTER', 'tgid != %s' % args.pid)
    self.bpf_text = bpf_text.replace('FILTER', '0')
    self.quanta_runtime_data = list[QuantaRuntimeData]()

  def load(self):
    # this would be nice but does not work with only capabilities: CAP_BPF,CAP_SYS_ADMIN
    #is_support_raw_tp = BPF.support_raw_tracepoint()
    #if not is_support_raw_tp:
    #  raise NotImplementedError()
    self.bpf = BPF(text = self.bpf_text)
    self.bpf["quanta_runtimes"].open_perf_buffer(self._event_handler)
    print("Quanta Runtimes BPF program loaded")

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
        quanta_end_uptime_us=event.quanta_end_uptime_us,
        quanta_run_length_us=event.quanta_run_length_us,
      )
    )