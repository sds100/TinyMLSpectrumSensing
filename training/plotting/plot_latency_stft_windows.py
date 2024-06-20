import matplotlib.pyplot as plt
import numpy as np

windows = [64, 128, 256, 512, 1024]
latencies = [0, 0, 0, 0, 0]

# Number of categories
N = len(windows)
xs = np.arange(N)  # the x locations for the groups
width = 0.35  # the width of the bars


plt.figure(figsize=(5, 4), dpi=160)
plt.bar(xs, latencies, width, label='Latency (ms)')
plt.xlabel('Number of windows')
plt.ylabel('Latency (ms)')
plt.xticks(xs, labels=list(map(str, windows)))

# Display the plot
plt.savefig("../output/figures/stft-latencies.png")
plt.show()
