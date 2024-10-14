# from matplotlib import pyplot as plt
import polars as pl

from data_schema.schema import (
    CollectionGraph,
    CollectionTable,
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
        return []

    # the raw data is a cumulative representation, this returns the deltas
    def as_pdf(self) -> pl.DataFrame:
        by_cpu_dtlb_dfs = self.filtered_table().select([
            "cpu",
            "ts_uptime_us",
            "cumulative_tlb_misses",
            "pmu_enabled_time_us",
            "pmu_running_time_us",
            "dtlb_event",
            "itlb_event",
        ]).filter(
            pl.col("dtlb_event")
        ).group_by("cpu")
        by_cpu_dtlb_pdf_dfs = [
            by_cpu_dtlb_df.lazy().sort("ts_uptime_us").with_columns(
                pl.col("cumulative_tlb_misses").shift(1, fill_value=0).alias("cumulative_tlb_misses_shifted"),
                pl.col("pmu_running_time_us").shift(1, fill_value=0).alias("pmu_running_time_us_shifted"),
                pl.col("pmu_enabled_time_us").shift(1, fill_value=0).alias("pmu_enabled_time_us_shifted"),
            ).with_columns(
                (pl.col("cumulative_tlb_misses") - pl.col("cumulative_tlb_misses_shifted")).alias("tlb_misses_raw"),
                (pl.col("pmu_running_time_us") - pl.col("pmu_running_time_us_shifted")).alias("span_duration_us"),
            ).with_columns(
                (
                    (
                        pl.col("span_duration_us")
                    ) / (
                        pl.col("pmu_enabled_time_us") - pl.col("pmu_enabled_time_us_shifted")
                    )
                ).alias("sampling_scaling"),
            ).with_columns(
                (pl.col("tlb_misses_raw") * pl.col("sampling_scaling")).alias("tlb_misses"),
            ).select([
                "cpu",
                "ts_uptime_us",
                "span_duration_us",
                "tlb_misses",
                "dtlb_event",
                "itlb_event",
            ])
            for _, by_cpu_dtlb_df in by_cpu_dtlb_dfs
        ]
        dtlb_pdf_df = pl.concat(by_cpu_dtlb_pdf_dfs).collect()
        return dtlb_pdf_df
