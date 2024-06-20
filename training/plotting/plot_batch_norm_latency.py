import numpy as np
from matplotlib import pyplot as plt

no_batch_norm_latency = [51, 61, 82, 153]
batch_norm_latency = [70, 95, 149, 287]

x_filters = [1, 2, 4, 8]
x_axis = np.arange(len(x_filters))
y_ticks = np.arange(300, step=50)

bar_width = 0.35
plt.figure(figsize=(5, 4), dpi=160)
plt.bar(x=x_axis,
        width=bar_width,
        height=batch_norm_latency,
        tick_label=x_filters,
        label="Batch norm.")

plt.bar(x=x_axis + bar_width,
        width=bar_width,
        height=no_batch_norm_latency,
        tick_label=x_filters,
        label="No batch norm.",
        color="orange")

plt.yticks(y_ticks)
plt.ylabel("Latency (ms)")
plt.xlabel("Number of filters per convolutional layer")
plt.legend()
plt.savefig("../output/figures/batch-norm-latency.png")
plt.show()

mean_difference = np.mean(np.array(batch_norm_latency) - np.array(no_batch_norm_latency))
mean_percent_difference = 1 - np.mean(np.array(no_batch_norm_latency) / np.array(batch_norm_latency))
print(f"Mean difference = {mean_difference} ms")
print(f"Mean % difference = {mean_percent_difference}")
