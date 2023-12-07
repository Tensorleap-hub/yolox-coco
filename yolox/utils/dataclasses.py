from dataclasses import dataclass
from typing import Tuple


@dataclass
class RescaleMetadata:
    original_shape: Tuple[int, int]
    scale_factor_h: float
    scale_factor_w: float


@dataclass
class PaddingCoordinates:
    top: int
    bottom: int
    left: int
    right: int
