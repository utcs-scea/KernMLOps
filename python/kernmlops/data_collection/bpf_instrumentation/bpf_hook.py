"""Abstract definition of a BPF program."""

from typing import Mapping

import polars as pl
from typing_extensions import Protocol


class BPFProgram(Protocol):
  """Loadable BPF program that returns performance data."""

  @classmethod
  def name(cls) -> str: ...

  def load(self) -> None: ...

  def poll(self) -> None: ...

  def data(self) -> pl.DataFrame: ...

  def clear(self): ...

  def pop_data(self) -> pl.DataFrame: ...

  @classmethod
  def plot(cls,
    collections_dfs: Mapping[str, pl.DataFrame],
    collection_id: str | None = None
  ) -> None: ...
