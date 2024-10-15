# from matplotlib import pyplot as plt
import plotext as plt
import polars as pl

from data_schema.schema import (
    CollectionData,
    CollectionGraph,
    CollectionTable,
    cumulative_pma_as_pdf,
)


class TLBPerfTable(CollectionTable):

    @classmethod
    def name(cls) -> str:
        return "tlb_perf"

    @classmethod
    def schema(cls) -> pl.Schema:
        return pl.Schema({
            "dtlb_event": pl.Boolean(),
            "itlb_event": pl.Boolean(),
            "cpu": pl.Int64(),
            "ts_uptime_us": pl.Int64(),
            "collection_id": pl.String(),
            "cumulative_tlb_misses": pl.Int64(),
            "pmu_enabled_time_us": pl.Int64(),
            "pmu_running_time_us": pl.Int64(),
        })

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> "TLBPerfTable":
        return TLBPerfTable(table=table.cast(cls.schema(), strict=True))  # pyright: ignore [reportArgumentType]

    def __init__(self, table: pl.DataFrame):
        self._table = table

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    def filtered_table(self) -> pl.DataFrame:
        return self.table

    def graphs(self) -> list[type[CollectionGraph]]:
        return [TLBPerfGraph]

    # the raw data is a cumulative representation, this returns the deltas
    def as_pdf(self) -> pl.DataFrame:
        dtlb_df = self.filtered_table().select([
            "cpu",
            "ts_uptime_us",
            "cumulative_tlb_misses",
            "pmu_enabled_time_us",
            "pmu_running_time_us",
            "dtlb_event",
            "itlb_event",
        ]).filter(
            pl.col("dtlb_event")
        )
        return cumulative_pma_as_pdf(
            dtlb_df,
            counter_column="cumulative_tlb_misses",
            counter_column_rename="tlb_misses",
        )


class TLBPerfGraph(CollectionGraph):

    @classmethod
    def with_collection(cls, collection_data: "CollectionData") -> "CollectionGraph | None":
        tlb_perf_table = collection_data.get(TLBPerfTable)
        if tlb_perf_table is not None:
            return TLBPerfGraph(
                collection_data=collection_data,
                tlb_perf_table=tlb_perf_table
            )
        return None

    @classmethod
    def base_name(cls) -> str:
        return "TLB Performance"

    def __init__(
        self,
        collection_data: CollectionData,
        tlb_perf_table: TLBPerfTable,
    ):
        self.collection_data = collection_data
        self._tlb_perf_table = tlb_perf_table

    def name(self) -> str:
        return f"{self.base_name()} for Collection {self.collection_data.id}"

    def x_axis(self) -> str:
        return "Benchmark Runtime (sec)"

    def y_axis(self) -> str:
        return "TLB Misses/msec"

    def plot(self) -> None:
        dtlb_df = self._tlb_perf_table.as_pdf()
        # TODO(Patrick): fix tlb misses to scale by span time
        start_uptime_sec = self.collection_data.start_uptime_sec

        # group by and plot by cpu
        dtlb_df_by_cpu = dtlb_df.group_by("cpu")
        for cpu, dtlb_df_group in dtlb_df_by_cpu:
            plt.plot(
                (
                    (dtlb_df_group.select("ts_uptime_us") / 1_000_000.0) - start_uptime_sec
                ).to_series().to_list(),
                (
                    dtlb_df_group.select("tlb_misses") / (
                        dtlb_df_group.select("span_duration_us") / 1_000.0
                    )
                ).to_series().to_list(),
                label=f"CPU {cpu[0]}",
            )

    def plot_trends(self) -> None:
        pass
