from dataclasses import dataclass
from typing import List

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Result:
    snr: int
    label_names: List[str]

    # The labels and predictions should be the same length so a confusion matrix
    # can be made.
    full_model_labels: List[List[int]]
    full_model_predictions: List[List[int]]

    # The labels and predictions should be the same length so a confusion matrix
    # can be made.
    lite_model_labels: List[List[int]]
    lite_model_predictions: List[List[int]]
    lite_model_size: int

    # These are the results for the model without quantization
    lite_model_no_quant_labels: List[List[int]]
    lite_model_no_quant_predictions: List[List[int]]
    lite_no_quant_model_size: int
