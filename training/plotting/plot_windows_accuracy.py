import matplotlib.pyplot as plt
import numpy as np

from plotting.plotting_utils import read_results, calc_accuracy

windows = [64, 128, 256, 512, 1024]
snrs = [0, 5, 10, 15, 20, 25, 30]

x_labels = list(map(str, snrs))

xs = np.arange(len(snrs))  # the x locations for the groups

plt.figure(figsize=(5, 4), dpi=160)
plt.rc('grid', linestyle=":")
plt.grid()

markers = ["o", "s", "D", "^", "X"]

for (marker, w) in zip(markers, windows):
    results = read_results(f"../output/results-windows-{w}.json")

    accuracies = list(
        map(lambda snr: calc_accuracy(results[snr].get_all_lite_model_labels(),
                                      results[snr].get_all_lite_model_predictions()),
            snrs))

    plt.plot(accuracies, label=f"{w} windows", marker=marker)

plt.xlabel('SNR (dB)')
plt.ylabel('Accuracy')
plt.xticks(ticks=xs, labels=list(map(str, x_labels)))
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.legend(loc='lower right')
plt.tight_layout()

plt.savefig("../output/figures/windows-accuracy.png")
plt.show()
