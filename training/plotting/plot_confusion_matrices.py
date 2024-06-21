from typing import List

import numpy as np
from matplotlib import pyplot as plt
from seaborn import heatmap
from sklearn.metrics import confusion_matrix

from plotting.plotting_utils import read_results, calc_accuracy
from spectrum_painting_result import SpectrumPaintingResult

results = read_results("../output/results-windows-64.json")

confusion_matrix_snr = 30
confusion_matrix_result: SpectrumPaintingResult = results[confusion_matrix_snr]


def plot_confusion_matrix(y_test,
                          y_predictions,
                          label_names: List[str]):
    cm = confusion_matrix(y_test, y_predictions)
    row_sums = cm.sum(axis=1)[:, np.newaxis]

    cm = (cm.astype(np.float32) / row_sums)
    cm = cm.round(2)

    plt.figure(figsize=(4, 4), dpi=160)
    plot = heatmap(cm, xticklabels=label_names, yticklabels=label_names, annot=True, cmap='Blues', cbar=False)
    plot.get_figure()
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.show()


def plot_confusion_matrix_standard_deviation(y_test: List[List[int]],
                                             y_predictions: List[List[int]],
                                             label_names: List[str]):
    cms = []

    for i in range(len(y_predictions)):
        cm = confusion_matrix(y_test[i], y_predictions[i])
        cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
        cms.append(cm)

    plt.figure(dpi=160)
    heatmap(np.std(cms, axis=0), cmap='Blues', annot=True, xticklabels=label_names, yticklabels=label_names,
            cbar=False)
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.show()


print("Full model")
plot_confusion_matrix(confusion_matrix_result.get_all_full_model_labels(),
                      confusion_matrix_result.get_all_full_model_predictions(),
                      confusion_matrix_result.label_names)

# plot_confusion_matrix_standard_deviation(confusion_matrix_result.get_all_full_model_labels(),
#                       confusion_matrix_result.get_all_full_model_predictions(),
#                       confusion_matrix_result.label_names)

print("Lite model")
plot_confusion_matrix(confusion_matrix_result.get_all_lite_model_labels(),
                      confusion_matrix_result.get_all_lite_model_predictions(),
                      confusion_matrix_result.label_names)

# Plot all confusion matrices
for (snr, result) in results.items():
    print(f"SNR {snr}")

    y_test = result.get_all_lite_model_labels()
    predictions = result.get_all_lite_model_predictions()

    avg_accuracy = calc_accuracy(y_test, predictions)
    print(f"Accuracy = {avg_accuracy}")

    plot_confusion_matrix(predictions, y_test, result.label_names)
    # plot_confusion_matrix_standard_deviation(result.full_model_predictions, result.full_model_labels, result.label_names)
