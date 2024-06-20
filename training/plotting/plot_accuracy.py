import numpy as np
from matplotlib import pyplot as plt

from plotting.plotting_utils import read_results, calc_accuracy

results = read_results("../output/results-filters-2.json")

baseline_accuracy = {
    0: 0.37,
    5: 0.45,
    10: 0.57,
    15: 0.7,
    20: 0.82,
    25: 0.90,
    30: 0.93
}

y_full_accuracy = []
y_lite_accuracy = []
y_baseline_accuracy = []
x_snr = []

for (snr, result) in results.items():
    x_snr.append(snr)

    if baseline_accuracy.__contains__(snr):
        y_baseline_accuracy.append(baseline_accuracy[snr])
    else:
        y_baseline_accuracy.append(0)

    y_test = result.get_all_lite_model_labels()
    predictions = result.get_all_lite_model_predictions()

    full_model_avg_accuracy = calc_accuracy(result.get_all_full_model_labels(), result.get_all_full_model_predictions())
    y_full_accuracy.append(full_model_avg_accuracy)

    lite_model_avg_accuracy = calc_accuracy(result.get_all_lite_model_labels(), result.get_all_lite_model_predictions())
    y_lite_accuracy.append(lite_model_avg_accuracy)

y_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
x_axis = np.arange(len(x_snr))

bar_width = 0.35
plt.figure(figsize=(5, 4), dpi=160)
plt.bar(x=x_axis,
        width=bar_width,
        height=y_baseline_accuracy,
        tick_label=x_snr,
        label="Baseline")

plt.bar(x=x_axis + bar_width,
        width=bar_width,
        height=y_lite_accuracy,
        tick_label=x_snr,
        label="Our results",
        color="orange")
plt.yticks(y_ticks)
plt.ylabel("Accuracy")
plt.xlabel("SNR (dB)")
plt.legend()
plt.savefig("../output/figures/bar-chart-accuracy.png")
plt.show()

average_improvement = np.mean(np.array(y_lite_accuracy) - np.array(y_baseline_accuracy))
print(f"Average improvement = {average_improvement}")