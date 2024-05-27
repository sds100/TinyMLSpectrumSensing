from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import train_test_split

import spectrum_painting as sp
from training.spectrogram import Spectrogram, split_spectrogram


@dataclass
class SpectrumPaintingTrainingOptions:
    spectrogram_length: int
    downsample_resolution: int
    k: int
    l: int
    d: int
    color_depth: int


@dataclass
class SpectrumPaintingTrainTestSets:
    x_train_augmented: List[npt.NDArray[np.uint8]]
    x_train_painted: List[npt.NDArray[np.uint8]]
    y_train: npt.NDArray[np.uint8]
    x_test_augmented: List[npt.NDArray[np.uint8]]
    x_test_painted: List[npt.NDArray[np.uint8]]
    y_test: npt.NDArray[np.uint8]
    label_names: List[str]


def create_spectrum_painting_train_test_sets(spectrograms: Dict[str, Spectrogram],
                                             options: SpectrumPaintingTrainingOptions):
    x_train_augmented = []
    x_train_painted = []
    labels: List[np.uint8] = []
    label_names: List[str] = []

    for (class_index, (label, spec)) in enumerate(spectrograms.items()):
        # middle: int = len(spec.values) // 2
        # start_freq: int = middle - 64
        # end_freq: int = middle + 64
        # 
        # spec = sp.take_frequencies(spec, start_freq, end_freq)

        slices = split_spectrogram(spec, duration=options.spectrogram_length)

        downsampled_slices = [sp.downsample_spectrogram(s.values, options.downsample_resolution) for s in slices]
        augmented_slices = [sp.augment_spectrogram(s, options.k, options.l, options.d) for s in downsampled_slices]

        for s in augmented_slices:
            x_train_augmented.append(sp.digitize_spectrogram(s, options.color_depth))

        painted_slices = [sp.paint_spectrogram(original, augmented) for (original, augmented) in
                          list(zip(downsampled_slices, augmented_slices))]

        for s in painted_slices:
            x_train_painted.append(sp.digitize_spectrogram(s, options.color_depth))

        for i in range(len(slices)):
            labels.append(class_index)

        label_names.append(label)

    x_train_combined = np.stack((x_train_augmented, x_train_painted), axis=3)

    x_train, x_test, y_train, y_test = train_test_split(x_train_combined, labels, test_size=0.3, random_state=42)

    # for tensorflow it must be uint8 and not a Python int.
    y_train = np.asarray(y_train, dtype=np.uint8)
    y_test = np.asarray(y_test, dtype=np.uint8)

    x_train_augmented = x_train[:, :, :, 0]
    x_test_augmented = x_test[:, :, :, 0]

    x_train_painted = x_train[:, :, :, 1]
    x_test_painted = x_test[:, :, :, 1]

    return SpectrumPaintingTrainTestSets(
        x_train_augmented,
        x_train_painted,
        y_train,
        x_test_augmented,
        x_test_painted,
        y_test,
        label_names
    )
