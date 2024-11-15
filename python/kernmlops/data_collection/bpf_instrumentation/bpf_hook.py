"""Abstract definition of a BPF program."""

import time

import polars as pl
from data_schema import UPTIME_TIMESTAMP, CollectionTable
from typing_extensions import Final, Protocol

POLL_TIMEOUT_MS: Final[int] = 10


class BPFProgram(Protocol):
  """Loadable BPF program that returns performance data."""
  data_dfs = list[pl.DataFrame]()

  @classmethod
  def name(cls) -> str: ...

  def load(self, collection_id: str) -> None: ...

  def poll(self) -> None: ...

  def close(self) -> None: ...

  def collection_table_types(cls) -> list[type[CollectionTable]]: ...

  def parse_data(self) -> list[CollectionTable]: ...

  def cache_data(self) -> list[CollectionTable]:
    new_data_dfs = self.pop_data()
    if not self.data_dfs:
      self.data_dfs = new_data_dfs
    else:
      data_dfs = []
      for old_df, new_df in zip(self.data_dfs, new_data_dfs, strict=True):
        if not new_df:
          data_dfs.append(old_df)
        else:
          data_dfs.append(pl.concat([old_df, new_df]))
      self.data_dfs = data_dfs
    return self.data_dfs

  def last_k_ms(self, ms: int) -> list[CollectionTable]:
    now_uptime_us=int(time.clock_gettime_ns(time.CLOCK_BOOTTIME) / 1000),
    last_k_us_ts = now_uptime_us - ms * 1_000
    return [
      data_df.filter(
        pl.col(UPTIME_TIMESTAMP) > last_k_us_ts
      )
      for data_df in self.data_dfs
    ]

  def last_k_data(self, k: int) -> list[CollectionTable]:
    return [
      data_df.tail(k)
      for data_df in self.data_dfs
    ]

  def last_data(self) -> list[CollectionTable]:
    return self.last_data(1)

  def clear(self): ...

  def pop_data(self) -> list[CollectionTable]: ...
