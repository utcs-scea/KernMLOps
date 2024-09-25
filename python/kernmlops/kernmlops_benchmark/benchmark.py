"""Abstract definition of a benchmark."""

from typing_extensions import Protocol


class Benchmark(Protocol):
  """Runnable benchmark that terminates naturally in finite time."""

  @classmethod
  def name(cls) -> str: ...

  def is_configured(self) -> bool:
    """Returns True if the environment has been setup to run the benchmark."""
    ...

  def setup(self) -> None: ...

  def run(self) -> None: ...

  def poll(self) -> bool:
    """Returns True when benchmark has is running."""
    ...

  def wait(self) -> None: ...

  def kill(self) -> None: ...


class FauxBenchmark(Benchmark):
  """Benchmark that does nothing and allows users to collect the running system's data."""

  @classmethod
  def name(cls) -> str:
    return "faux"

  def is_configured(self) -> bool:
    return True

  def setup(self) -> None:
    pass

  def run(self) -> None:
    pass

  def poll(self) -> bool:
    """This benchmark will never self terminate."""
    return True

  def wait(self) -> None:
    pass

  def kill(self) -> None:
    pass
