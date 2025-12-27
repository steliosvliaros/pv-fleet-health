from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    """Centralized paths (project-relative)."""
    root: Path

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def stage_dir(self) -> Path:
        return self.outputs_dir / "stages"

    @property
    def plots_dir(self) -> Path:
        return self.outputs_dir / "plots"

    def ensure(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
