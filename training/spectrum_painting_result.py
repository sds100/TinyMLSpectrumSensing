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

    def get_all_full_model_labels(self) -> List[int]:
        output: List[int] = []

        for result in self.full_model_results:
            output.extend(result.labels)

        return output

    def get_all_full_model_predictions(self) -> List[int]:
        output: List[int] = []

        for result in self.full_model_results:
            output.extend(result.predictions)

        return output

    def get_all_lite_model_labels(self) -> List[int]:
        output: List[int] = []

        for result in self.lite_model_results:
            output.extend(result.labels)

        return output

    def get_all_lite_model_predictions(self) -> List[int]:
        output: List[int] = []

        for result in self.lite_model_results:
            output.extend(result.predictions)

        return output
