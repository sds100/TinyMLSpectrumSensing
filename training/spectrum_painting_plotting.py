from typing import List

import numpy as np
import numpy.typing as npt
import seaborn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import History

from training.spectrogram import Spectrogram


def plot_spectrogram(spectrogram: Spectrogram):
    plt.pcolormesh(spectrogram.f, spectrogram.t, spectrogram.values.T, shading="nearest", cmap="viridis")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Time (s)")
    plt.title("Spectrogram")
    plt.colorbar(label="Magnitude (dB)")
    plt.show()


def plot_confusion_matrix(y_predictions: npt.NDArray[np.uint8],
                          y_test: npt.NDArray[np.uint8],
                          label_names: List[str]):
    cm = confusion_matrix(y_test, y_predictions)
    cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])

    plt.figure(dpi=80)
    plot = seaborn.heatmap(cm, xticklabels=label_names, yticklabels=label_names, annot=True, cmap='Blues')
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
    plt.plot(model_history.history['val_accuracy'], label='Validation accuracy')
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
