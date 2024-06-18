from dataclasses import dataclass
from typing import List

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ModelResult:
    labels: List[int]
    predictions: List[int]
    size: int


@dataclass_json
@dataclass
class SpectrumPaintingResult:
    snr: int
    label_names: List[str]

    full_model_results: List[ModelResult]
    lite_model_results: List[ModelResult]
