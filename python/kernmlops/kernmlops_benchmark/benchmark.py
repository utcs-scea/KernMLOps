"""Abstract definition of a benchmark."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from config import ConfigBase
from data_schema import GraphEngine
from typing_extensions import Protocol


@dataclass(frozen=True)
class GenericBenchmarkConfig(ConfigBase):
  benchmark: str = "faux"
  benchmark_dir: str = str(Path.home() / "kernmlops-benchmark")
  cpus: int = 0
  transparent_hugepages: Literal["always", "madvise", "never"] = "always"


@dataclass(frozen=True)
class FauxBenchmarkConfig(ConfigBase):
  pass


class Benchmark(Protocol):
  """Runnable benchmark that terminates naturally in finite time."""

  @classmethod
  def name(cls) -> str: ...

  @classmethod
  def default_config(cls) -> ConfigBase: ...

  def is_configured(self) -> bool:
    """Returns True if the environment has been setup to run the benchmark."""
    ...

  def setup(self) -> None: ...

  def run(self) -> None: ...

  def poll(self) -> int | None:
    """Returns None when benchmark is running, nonzero when crashed."""
    ...

  def wait(self) -> None: ...

  def kill(self) -> None: ...

  @classmethod
  def plot_events(cls, graph_engine: GraphEngine) -> None:
    """Given a collection of data, plot important events for this benchmark."""
    ...


class FauxBenchmark(Benchmark):
  """Benchmark that does nothing and allows users to collect the running system's data."""

  @classmethod
  def name(cls) -> str:
    return "faux"

  @classmethod
  def default_config(cls) -> ConfigBase:
    return FauxBenchmarkConfig()

  def is_configured(self) -> bool:
    return True

  def setup(self) -> None:
    pass

  def run(self) -> None:
    pass

  def poll(self) -> int | None:
    """This benchmark will never self terminate."""
    return None

  def wait(self) -> None:
    pass

  def kill(self) -> None:
    pass

  @classmethod
  def plot_events(cls, graph_engine: GraphEngine) -> None:
    pass
