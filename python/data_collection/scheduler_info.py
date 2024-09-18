from dataclasses import dataclass
from time import sleep

import polars as pl
from bcc import BPF


@dataclass(frozen=True)
class QuantaRuntimeData:
  cpu: int
  pid: int
  quanta_end_uptime_us: int
  quanta_run_length_us: int

bpf_text = open("bpf/sched_quanta_runtime.bpf.c", "r").read()
is_support_raw_tp = BPF.support_raw_tracepoint()

# code substitutions
if BPF.kernel_struct_has_field(b'task_struct', b'__state') == 1:
    bpf_text = bpf_text.replace('STATE_FIELD', '__state')
else:
    bpf_text = bpf_text.replace('STATE_FIELD', 'state')
# pid from userspace point of view is thread group from kernel pov
# bpf_text = bpf_text.replace('FILTER', 'tgid != %s' % args.pid)
bpf_text = bpf_text.replace('FILTER', '0')

bpf = BPF(text = bpf_text)
quanta_runtime_data = list[QuantaRuntimeData]()

# this would be nice but does not work with only capabilities: CAP_BPF,CAP_SYS_ADMIN
#if not is_support_raw_tp:
#  raise NotImplementedError()

def print_event(cpu, quanta_runtime_perf_event, size):
  event = bpf["quanta_runtimes"].event(quanta_runtime_perf_event)
  quanta_runtime_data.append(
    QuantaRuntimeData(
      cpu=cpu,
      pid=event.pid,
      quanta_end_uptime_us=event.quanta_end_uptime_us,
      quanta_run_length_us=event.quanta_run_length_us,
    )
  )

bpf["quanta_runtimes"].open_perf_buffer(print_event)
print("Quanta Runtimes BPF program loaded")

exiting = 0
while (1):
    try:
        bpf.perf_buffer_poll()
        sleep(1)
    except KeyboardInterrupt:
        break

sched_df = pl.DataFrame(quanta_runtime_data)
print(sched_df)
