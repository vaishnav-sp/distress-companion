"""Shared output contract for the future modality branches."""

from dataclasses import dataclass
from typing import Mapping, Protocol


@dataclass(frozen=True)
class BranchOutput:
    """Common result shape; Phase 1 does not populate model outputs."""

    features: Mapping[str, float] | None
    provisional_distress: float | None
    input_quality: bool


class BranchProcessor(Protocol):
    """Interface to be implemented by a face, behavior, or audio branch."""

    def process(self, input_data: object) -> BranchOutput:
        """Return extracted features and quality metadata for one input."""
        ...
