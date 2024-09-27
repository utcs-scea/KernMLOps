from typing import cast

import plotext as plt
import polars as pl

from data_schema.schema import (
    CollectionData,
    CollectionGraph,
    CollectionTable,
)


class QuantaRuntimeTable(CollectionTable):

    @classmethod
    def name(cls) -> str:
        return "quanta_runtime"

    @classmethod
    def schema(cls) -> pl.Schema:
        return pl.Schema()

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> "QuantaRuntimeTable":
        return QuantaRuntimeTable(table=table)

    def __init__(self, table: pl.DataFrame):
        self._table = table

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    def filtered_table(self) -> pl.DataFrame:
        # filter out invalid data points due to data loss
        initial_datapoints = len(self.table)
        max_run_length = 60_000
        quanta_df = self.table.filter(
          pl.col("quanta_run_length_us") < max_run_length
        )
        datapoints_removed = initial_datapoints - len(quanta_df)
        # TODO(Patrick): use logging
        print(f"Filtered out {datapoints_removed} datapoints with max run length {max_run_length}us")
        return quanta_df

    def graphs(self) -> list[type[CollectionGraph]]:
        return [QuantaRuntimeGraph]

    def total_runtime_us(self) -> int:
        """Returns the total amount of runtime recorded across all cpus."""
        return self.filtered_table().select(
            "quanta_run_length_us"
        ).sum()["quanta_run_length_us"].to_list()[0]
    def top_k_runtime(self, k: int) -> pl.DataFrame:
        """Returns the pids and execution time of the k processes with the most execution time."""
        return self.filtered_table().select(
            ["pid", "quanta_run_length_us"]
        ).group_by(
            "pid"
        ).sum().sort(
            "quanta_run_length_us", descending=True
        ).limit(k)


class QuantaRuntimeGraph(CollectionGraph):

    @classmethod
    def with_collection(cls, collection_data: "CollectionData") -> "CollectionGraph":
        return QuantaRuntimeGraph(collection_data=collection_data)

    @classmethod
    def _quanta_runtime_table_name(cls) -> str:
        return QuantaRuntimeTable.name()

    @classmethod
    def base_name(cls) -> str:
        return "Quanta Runtimes"

    def __init__(self, collection_data: CollectionData):
        self.collection_data = collection_data

    def name(self) -> str:
        return f"{self.base_name()} for Collection {self.collection_data.id}"

    def x_axis(self) -> str:
        return "Benchmark Runtime (sec)"

    def y_axis(self) -> str:
        return "Quanta Run Length (usec)"

    def valid(self) -> bool:
        return self._quanta_runtime_table_name() in self.collection_data.tables

    def plot(self) -> None:
        quanta_df = self.collection_data.tables[
            self._quanta_runtime_table_name()
        ].filtered_table()
        benchmark_start_time_sec = self.collection_data.benchmark_time_sec

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

    def plot_trends(self) -> None:
        quanta_table = cast(
            QuantaRuntimeTable,
            self.collection_data.tables[
                self._quanta_runtime_table_name()
            ]
        )
        quanta_df = quanta_table.filtered_table()
        benchmark_start_time_sec = self.collection_data.benchmark_time_sec
        collector_pid = self.collection_data.pid
        top_k = quanta_table.top_k_runtime(k=3)
        print(top_k)
        pids = top_k["pid"].to_list() + [collector_pid]

        for pid in pids:
            # Add trend of collector process to graph
            collector_runtimes = quanta_df.filter(
                pl.col("pid") == pid
            )
            plt.plot(
                (
                    (collector_runtimes.select("quanta_end_uptime_us") / 1_000_000) - benchmark_start_time_sec
                ).to_series().to_list(),
                collector_runtimes.select("quanta_run_length_us").to_series().to_list(),
                label="Collector Process" if collector_pid == pid else f"PID: {pid}",
                marker="braille",
            )

        print(f"Total processor time per cpu: {quanta_table.total_runtime_us() / 1_000_000.0 / self.collection_data.cpus }s")