from datetime import datetime
from pathlib import Path
from time import sleep

import data_collection
import polars as pl
from kernmlops_benchmark import Benchmark, BenchmarkNotConfiguredError


def poll_instrumentation(
  benchmark: Benchmark,
  bpf_programs: list[data_collection.bpf.BPFProgram],
  poll_rate: float = .5,
):
    while benchmark.poll():
        try:
            for bpf_program in bpf_programs:
                bpf_program.poll()
            sleep(poll_rate)
            # clean data when missed samples - or detect?
            # include collector pid
        except KeyboardInterrupt:
            benchmark.kill()
            break


def run_collect(
    *,
    output_dir: Path,
    benchmark: Benchmark,
    bpf_programs: list[data_collection.bpf.BPFProgram],
    poll_rate: float,
    verbose: bool
):
    if not benchmark.is_configured():
        raise BenchmarkNotConfiguredError(f"benchmark {benchmark.name()} is not configured")

    system_info = data_collection.machine_info().to_polars()
    system_info = system_info.unnest(system_info.columns)
    collection_id = system_info["collection_id"][0]

    benchmark.setup()

    for bpf_program in bpf_programs:
        bpf_program.load()
        if verbose:
            print(f"{bpf_program.name()} BPF program loaded")
    if verbose:
        print("Finished loading BPF programs")
    benchmark.run()
    if verbose:
        print(f"Started benchmark {benchmark.name()}")

    tick = datetime.now()
    poll_instrumentation(benchmark, bpf_programs, poll_rate=poll_rate)
    if verbose:
        print(f"Benchmark ran for {(datetime.now() - tick).total_seconds()}s")

    bpf_dfs = {
        bpf_program.name(): bpf_program.pop_data().with_columns(pl.lit(collection_id).alias("collection_id"))
        for bpf_program in bpf_programs
    }
    bpf_dfs["system_info"] = system_info
    for bpf_name, bpf_df in bpf_dfs.items():
        if verbose:
            print(f"{bpf_name}: {bpf_df}")
        Path(output_dir / bpf_name).mkdir(parents=True, exist_ok=True)
        bpf_df.write_parquet(output_dir / bpf_name / f"{collection_id}.{benchmark.name()}.parquet")
