import subprocess
from pathlib import Path

import psutil
from data_schema import demote

from kernmlops_benchmark.benchmark import Benchmark
from kernmlops_benchmark.errors import BenchmarkNotRunningError, BenchmarkRunningError


class LinuxBuildBenchmark(Benchmark):

    @classmethod
    def name(cls) -> str:
        return "linux-build"

    def __init__(self, benchmark_dir: Path, cpus: int | None = None):
        self.benchmark_dir = benchmark_dir / self.name()
        self.cpus = cpus or (3 * psutil.cpu_count(logical=False))
        self.process: subprocess.Popen | None = None

    def is_configured(self) -> bool:
        return self.benchmark_dir.is_dir()

    def setup(self) -> None:
        if self.process is not None:
            raise BenchmarkRunningError()
        subprocess.check_call(
            ["make", "-C", str(self.benchmark_dir), "clean"],
            preexec_fn=demote(),
            stdout=subprocess.DEVNULL,
        )
        subprocess.check_call(
            [
                "make",
                "-C",
                str(self.benchmark_dir / "../linux-kernel"),
                f"O={str(self.benchmark_dir)}",
                "defconfig",
            ],
            preexec_fn=demote(),
            stdout=subprocess.DEVNULL,
        )
        subprocess.check_call(
            ["bash", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches"],
            stdout=subprocess.DEVNULL,
        )

    def run(self) -> None:
        if self.process is not None:
            raise BenchmarkRunningError()
        jobs = f"-j{self.cpus}" if self.cpus else "-j"
        self.process = subprocess.Popen(
            ["make", "-C", str(self.benchmark_dir), jobs],
            preexec_fn=demote(),
            stdout=subprocess.DEVNULL,
        )


    def poll(self) -> int | None:
        if self.process is None:
            raise BenchmarkNotRunningError()
        return self.process.poll()

    def wait(self) -> None:
        if self.process is None:
            raise BenchmarkNotRunningError()
        self.process.wait()

    def kill(self) -> None:
        if self.process is None:
            raise BenchmarkNotRunningError()
        self.process.terminate()
