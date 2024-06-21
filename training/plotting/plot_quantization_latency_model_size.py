import matplotlib.pyplot as plt
import numpy as np

num_filters = [1, 2, 4, 8]
no_quant_size = [6860, 9180, 17372, 47964]
no_quant_latency = [31, 64, 144, 364]

quant_size = [7680, 8408, 10776, 19072]
quant_latency = [51, 61, 82, 153]

# Number of categories
xs = np.arange(len(num_filters))  # the x locations for the groups

x_axis = np.arange(len(num_filters))
y_ticks = np.arange(451, step=50)

bar_width = 0.35

plt.rc('axes', axisbelow=True)
plt.rc('grid', linestyle=":")
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), constrained_layout=True, dpi=160)

plt.subplot(1, 2, 1)
plt.grid(axis="y")
plt.bar(x=x_axis,
        width=bar_width,
        height=no_quant_latency,
        tick_label=num_filters,
        label="No quantization")

plt.bar(x=x_axis + bar_width,
        width=bar_width,
        height=quant_latency,
        tick_label=num_filters,
        label="Quantization",
        color="orange")

plt.yticks(y_ticks)
plt.ylabel("Latency (ms)")
plt.xlabel("Number of filters per convolutional layer")
plt.legend()

plt.rc('axes', axisbelow=True)
plt.rc('grid', linestyle=":")
plt.subplot(1, 2, 2)
plt.grid(axis="y")
y_ticks = np.arange(56, step=5)

plt.bar(x=x_axis,
        width=bar_width,
        height=np.array(no_quant_size) / 1000,
        tick_label=num_filters,
        label="No quantization")

plt.bar(x=x_axis + bar_width,
        width=bar_width,
        height=np.array(quant_size) / 1000,
        tick_label=num_filters,
        label="Quantization",
        color="orange")

plt.yticks(y_ticks)
plt.ylabel("Model size (KB)")
plt.xlabel("Number of filters per convolutional layer")
plt.legend()

plt.savefig("../output/figures/bar-chart-size-latency.png")
plt.show()
