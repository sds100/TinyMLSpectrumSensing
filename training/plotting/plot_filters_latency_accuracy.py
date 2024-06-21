import matplotlib.pyplot as plt
import numpy as np

num_filters = [1, 2, 4, 8]
inference_latencies = [51, 61, 82, 153]
inference_accuracies = [0.64, 0.94, 0.96, 0.96]

# Number of categories
N = len(num_filters)
xs = np.arange(N)  # the x locations for the groups
width = 0.35  # the width of the bars

plt.rc('axes', axisbelow=True)
plt.rc('grid', linestyle=":")
fig, ax1 = plt.subplots(figsize=(5, 4), dpi=160)
plt.grid(axis="y")

# Plotting latency bars
bars1 = ax1.bar(xs, inference_latencies, width, label='Latency (ms)', color='tomato')

# Creating a second y-axis for accuracy
ax2 = ax1.twinx()
bars2 = ax2.bar(xs + width, inference_accuracies, width, label='Accuracy', color='seagreen')

# Adding labels, title, and legend
ax1.set_xlabel('Number of filters per convolutional layer')
ax1.set_ylabel('Latency (ms)')
ax2.set_ylabel('Accuracy')
ax1.set_xticks(xs + width / 2)
ax1.set_xticklabels(num_filters)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left', bbox_to_anchor=(0, 1), frameon=True)

# Display the plot
plt.savefig("../output/figures/filters-latency-accuracy.png")
plt.show()
