import numpy as np
from matplotlib import pyplot as plt

from plotting.plotting_utils import read_results, calc_accuracy

results = read_results("results-10-iterations.json")

baseline_accuracy = {
    0: 0.37,
    5: 0.45,
    10: 0.57,
    15: 0.7,
    20: 0.82,
    30: 0.93
}
our_accuracy = []

y_full_accuracy = []
y_lite_accuracy = []
x_snr = []

for (snr, result) in results.items():
    x_snr.append(snr)

    y_test = result.get_all_lite_model_labels()
    predictions = result.get_all_lite_model_predictions()

    full_model_avg_accuracy = calc_accuracy(result.get_all_full_model_labels(), result.get_all_full_model_predictions())
    y_full_accuracy.append(full_model_avg_accuracy)

    lite_model_avg_accuracy = calc_accuracy(result.get_all_lite_model_labels(), result.get_all_lite_model_predictions())
    y_lite_accuracy.append(lite_model_avg_accuracy)

for snr in baseline_snrs:
    result = results[snr]
    accuracy = calc_accuracy(result.get_all_lite_model_labels(), result.get_all_lite_model_predictions())
    our_accuracy.append(accuracy)

y_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

plt.figure(figsize=(4, 4), dpi=160)
plt.bar(x=range(len(x_snr)),
        height=y_lite_accuracy,
        tick_label=x_snr)
plt.yticks(y_ticks)
plt.ylabel("Accuracy")
plt.xlabel("SNR (dB)")
plt.show()

x_axis = np.arange(len(baseline_snrs))
bar_width = 0.4

plt.figure(figsize=(4, 4), dpi=160)
plt.bar(x=x_axis - 0.2,
        width=bar_width,
        height=baseline_accuracy,
        tick_label=baseline_snrs,
        label="Baseline")

plt.bar(x=x_axis + 0.2,
        width=bar_width,
        height=our_accuracy,
        tick_label=baseline_snrs,
        label="Our results")

plt.legend()
plt.yticks(y_ticks)
plt.ylabel("Accuracy")
plt.xlabel("SNR (dB)")
plt.show()
