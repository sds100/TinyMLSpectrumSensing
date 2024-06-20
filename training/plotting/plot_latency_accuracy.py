import matplotlib.pyplot as plt
import numpy as np

num_filters = [1, 2, 4, 8]
inference_latencies = [70, 95, 149, 287]
inference_accuracies = [0.74, 0.95, 0.97, 0.94]

plt.figure(figsize=(3, 3), dpi=160)
fig, ax = plt.subplots()

ax.bar(x=range(len(num_filters)),
       height=inference_latencies,
       tick_label=num_filters)
ax.set_ylabel("Inference latency (ms)")
ax.set_xlabel("Number of filters per convolutional layer")

ax2 = ax.twinx()
ax2.bar(x=range(len(num_filters)),
        height=inference_accuracies,
        tick_label=num_filters)
ax.set_ylabel("Accuracy")

plt.show()


categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
latency = [100, 150, 200, 250]  # Latency in milliseconds
accuracy = [0.9, 0.85, 0.78, 0.92]  # Accuracy as a fraction

# Number of categories
N = len(categories)
ind = np.arange(N)  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax1 = plt.subplots()

# Plotting latency bars
bars1 = ax1.bar(ind, latency, width, label='Latency (ms)', color='b')

# Creating a second y-axis for accuracy
ax2 = ax1.twinx()
bars2 = ax2.bar(ind + width, accuracy, width, label='Accuracy', color='g')

# Adding labels, title, and legend
ax1.set_xlabel('Categories')
ax1.set_ylabel('Latency (ms)', color='b')
ax2.set_ylabel('Accuracy', color='g')
ax1.set_title('Latency and Accuracy by Category')
ax1.set_xticks(ind + width / 2)
ax1.set_xticklabels(categories)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Display the plot
plt.show()

