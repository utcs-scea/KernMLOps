"""Module for maintaining different BPF hooks/instrumentation."""

from functools import cache

from data_collection.bpf_instrumentation.bpf_hook import BPFProgram
from data_collection.bpf_instrumentation.file_data_hook import FileDataBPFHook
from data_collection.bpf_instrumentation.memory_usage_hook import MemoryUsageHook
from data_collection.bpf_instrumentation.process_metadata_hook import (
    ProcessMetadataHook,
)
from data_collection.bpf_instrumentation.quanta_runtime_hook import QuantaRuntimeBPFHook
from data_collection.bpf_instrumentation.tlb_perf_hook import (
    CustomHWConfigManager,
    TLBPerfBPFHook,
)

all_hooks = [
    FileDataBPFHook,
    MemoryUsageHook,
    ProcessMetadataHook,
    QuantaRuntimeBPFHook,
    TLBPerfBPFHook,
]


@cache
def hooks() -> list[BPFProgram]:
    return [
        ProcessMetadataHook(),
        FileDataBPFHook(),
        MemoryUsageHook(),
        QuantaRuntimeBPFHook(),
        TLBPerfBPFHook(),
    ]


__all__ = [
    "all_hooks",
    "hooks",
    "BPFProgram",
    "CustomHWConfigManager",
    "QuantaRuntimeBPFHook",
]
