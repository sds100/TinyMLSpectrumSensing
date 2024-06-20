import json
from typing import List, Dict

import numpy as np

from spectrum_painting_result import SpectrumPaintingResult


def read_results(file_name: str) -> Dict[int, SpectrumPaintingResult]:
    result_list: List[SpectrumPaintingResult] = []
    results: Dict[int, SpectrumPaintingResult] = {}

    with open(file_name, "r") as f:
        result_list = json.load(f)["results"]
        result_list = [SpectrumPaintingResult.from_dict(r) for r in result_list]
        for result in result_list:
            results[result.snr] = result

    return results


def calc_accuracy(y_test, predictions) -> float:
    return np.mean(np.asarray(y_test) == np.asarray(predictions))
