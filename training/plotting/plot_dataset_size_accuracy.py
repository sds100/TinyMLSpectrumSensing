from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from plotting.plotting_utils import read_results, calc_accuracy
from spectrum_painting_result import SpectrumPaintingResult

num_spectrograms = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

results: List[Dict[int, SpectrumPaintingResult]] = list(
    map(lambda n: read_results(f"../output/results-specs-{n}.json"), num_spectrograms))

accuracies_snr_30 = list(
    map(lambda r: calc_accuracy(r[30].get_all_lite_model_labels(), r[30].get_all_lite_model_predictions()), results))

accuracies_snr_20 = list(
    map(lambda r: calc_accuracy(r[20].get_all_lite_model_labels(), r[20].get_all_lite_model_predictions()), results))

accuracies_snr_10 = list(
    map(lambda r: calc_accuracy(r[10].get_all_lite_model_labels(), r[10].get_all_lite_model_predictions()), results))

accuracies_snr_0 = list(
    map(lambda r: calc_accuracy(r[0].get_all_lite_model_labels(), r[0].get_all_lite_model_predictions()), results))


def calculate_training_set_size(specs_per_class: int) -> int:
    # multiply by number of classes and SNRs
    return specs_per_class * 7 * 7


x_labels = list(map(calculate_training_set_size, num_spectrograms))
# x_labels = num_spectrograms

xs = np.arange(len(num_spectrograms))  # the x locations for the groups

plt.figure(figsize=(5, 4), dpi=160)
plt.plot(xs, accuracies_snr_30)
plt.plot(xs, accuracies_snr_20)
plt.plot(xs, accuracies_snr_10)
plt.plot(xs, accuracies_snr_0)

plt.xlabel('Training set size')
plt.ylabel('Accuracy')
plt.xticks(ticks=xs, labels=list(map(str, x_labels)), rotation=45)
plt.yticks(np.arange(0, 1.1, step=0.1))

plt.tight_layout()
# Display the plot
plt.savefig("../output/figures/ablation-study.png")
plt.show()
