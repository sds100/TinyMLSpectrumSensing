import matplotlib.pyplot as plt
import numpy as np

windows = [64, 128, 256, 512, 1024]
latencies = [34, 49, 77, 132, 245]

x_labels = list(map(str, windows))

xs = np.arange(len(windows))  # the x locations for the groups

plt.rc('axes', axisbelow=True)
plt.rc('grid', linestyle=":")
plt.figure(figsize=(5, 4), dpi=160)
plt.grid(axis="y", )
plt.bar(xs, latencies)

plt.xlabel('Number of STFT windows')
plt.ylabel('Latency (ms)')
plt.xticks(ticks=xs, labels=list(map(str, windows)))
plt.yticks(np.arange(0, 301, step=50))

# Display the plot
plt.savefig("../output/figures/windows-latency.png")
plt.show()
