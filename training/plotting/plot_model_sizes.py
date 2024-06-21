import matplotlib.pyplot as plt
import numpy as np

from plotting.plotting_utils import read_results

num_filters = ["1", "2", "4", "8"]

snr = 30
model_sizes = [
    read_results("../output/results-filters-1.json")[snr].lite_model_results[0].size,
    read_results("../output/results-filters-2.json")[snr].lite_model_results[0].size,
    read_results("../output/results-filters-4.json")[snr].lite_model_results[0].size,
    read_results("../output/results-filters-8.json")[snr].lite_model_results[0].size
]

xs = np.arange(len(num_filters))  # the x locations for the groups

plt.rc('axes', axisbelow=True)
plt.rc('grid', linestyle=":")
plt.figure(figsize=(5, 4), dpi=160)
plt.grid(axis="y")
plt.bar(xs, np.array(model_sizes) / 1000)

plt.xlabel('Number of filters per convolutional layer')
plt.ylabel('Model size (KB)')
plt.xticks(ticks=xs, labels=num_filters)
plt.yticks(np.arange(0, 26, step=5, dtype=int))

# Display the plot
plt.savefig("../output/figures/model-sizes.png")
plt.show()
