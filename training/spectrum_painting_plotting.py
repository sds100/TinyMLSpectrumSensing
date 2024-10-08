from typing import List

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import History


def plot_spectrogram(spectrogram: npt.NDArray, title: str = "Spectrogram"):
    plt.figure(figsize=(3, 3))
    plt.imshow(spectrogram, cmap='viridis')
    plt.title(title)
    plt.colorbar(label='Magnitude')
    plt.xlabel('Frequency')
    plt.ylabel('Time')
    plt.title(title)
    plt.show()


def plot_confusion_matrix(y_predictions,
                          y_test,
                          label_names: List[str]):
    cm = confusion_matrix(y_test, y_predictions)
    cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])

    plt.figure(dpi=80)
    plot = heatmap(cm, xticklabels=label_names, yticklabels=label_names, annot=True, cmap='Blues')
    plot.get_figure()
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.show()


def plot_train_images(images: List[npt.NDArray], labels: List[int], label_names: List[str], snr_list: List[int]):
    plt.figure(figsize=(15, 15))
    for i in range(min(len(images), 50)):
        image = images[i]
        plt.subplot(6, 10, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap="viridis")
        plt.xlabel(f"{label_names[labels[i]]} {snr_list[i]}")
    plt.show()


def plot_model_accuracy_epochs(model_history: History):
    plt.plot(model_history.history['accuracy'], label='Train accuracy')
    plt.plot(model_history.history['val_accuracy'], label='Test accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='upper right')
    plt.show()


def plot_model_loss(model_history: History):
    plt.plot(model_history.history['loss'], label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, 2])
    plt.show()
