

# from matplotlib import pyplot as plt
import polars as pl

from data_schema.schema import (
    CollectionGraph,
    CollectionTable,
)


class FileDataTable(CollectionTable):

    @classmethod
    def name(cls) -> str:
        return "file_data"

    @classmethod
    def schema(cls) -> pl.Schema:
        return pl.Schema({
            "cpu": pl.Int64(),
            "pid": pl.Int64(),
            "tgid": pl.Int64(),
            "ts_uptime_us": pl.Int64(),
            "file_inode": pl.Int64(),
            "file_size_bytes": pl.Int64(),
            "file_name": pl.String(),
            "collection_id": pl.String(),
        })

    @classmethod
    def from_df(cls, table: pl.DataFrame) -> "FileDataTable":
        return FileDataTable(table=table.cast(cls.schema(), strict=True))  # pyright: ignore [reportArgumentType]

    def __init__(self, table: pl.DataFrame):
        self._table = table

    @property
    def table(self) -> pl.DataFrame:
        return self._table

    def filtered_table(self) -> pl.DataFrame:
        return self.table

    def graphs(self) -> list[type[CollectionGraph]]:
        return []

    def total_files_opened(self) -> int:
        """Returns the total amount of unique files opened across all cpus."""
        return len(self.filtered_table().select("file_inode").unique("file_inode"))
