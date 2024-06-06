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
    train_snr: List[int]

    x_test_augmented: List[npt.NDArray[np.uint8]]
    x_test_painted: List[npt.NDArray[np.uint8]]
    y_test: npt.NDArray[np.uint8]
    test_snr: List[int]

    label_names: List[str]


def create_augmented_painted_images(spectrogram: npt.NDArray,
                                    options: SpectrumPaintingTrainingOptions) -> (
        npt.NDArray[np.uint8], npt.NDArray[np.uint8]):
    downsampled = sp.downsample_spectrogram(spectrogram, options.downsample_resolution)
    augmented = sp.augment_spectrogram(downsampled, options.k, options.l, options.d)
    digitized_augmented = sp.digitize_spectrogram(augmented, options.color_depth)

    painted = sp.paint_spectrogram(downsampled, augmented)
    digitized_painted = sp.digitize_spectrogram(painted, options.color_depth)

    return digitized_augmented, digitized_painted


def create_spectrum_painting_train_test_sets(spectrograms: Dict[int, List[Spectrogram]],
                                             options: SpectrumPaintingTrainingOptions,
                                             test_size: float = 0.3) -> SpectrumPaintingTrainTestSets:
    """
    Create the training, test and label sets from a list of spectrograms.
    :param spectrograms: A dictionary that maps the class (Z, B, ZBW etc) to spectrograms with different
                        signal-to-noise ratios.
    :param options: The spectrum painting parameters.
    :param test_size: The proportion of the data to be in the test set.
    """
    digitized_augmented_slices: List[npt.NDArray[np.uint8]] = []
    digitized_painted_slices: List[npt.NDArray[np.uint8]] = []
    labels: List[int] = []
    label_names: List[str] = []
    snr_list: List[int] = []

    removed_image_count: int = 0

    for (snr_index, (snr, spectrogram_list)) in enumerate(spectrograms.items()):
        sliced_spectrograms: Dict[str, List[npt.NDArray]] = {}

        for spec in spectrogram_list:
            # Taking the middle of the spectrogram is not needed if you use
            # high D (step size) values. The reason why you may need it for small step
            # sizes is that for painting to remove the WiFi signals, they must fill the
            # entire width of the spectrogram.

            middle: int = len(spec.values) // 2
            start_freq: int = middle - 32
            end_freq: int = middle + 32

            spec = sp.take_frequencies(spec, start_freq, end_freq)

            slices = split_spectrogram(spec.values, duration=options.spectrogram_length)
            sliced_spectrograms[spec.label] = slices

        include_indices: List[int] = []

        for (i, s) in enumerate(sliced_spectrograms["B"]):
            (augmented, painted) = create_augmented_painted_images(s, options)

            mean_painted = np.mean(painted)
            max_painted = np.max(painted)

            if mean_painted < 1:
                removed_image_count += 1
            else:
                include_indices.append(i)

        for label_index, (label, slices) in enumerate(sliced_spectrograms.items()):
            for i in include_indices:
                s = slices[i]
                (augmented, painted) = create_augmented_painted_images(s, options)

                digitized_augmented_slices.append(augmented)
                digitized_painted_slices.append(painted)
                labels.append(label_index)
                snr_list.append(snr)

            label_names.append(label)
            
    print(f"Removed {removed_image_count} images that didn't actually contain any Bluetooth signals.")

    x_combined = np.stack((digitized_augmented_slices, digitized_painted_slices), axis=3)

    labels_snr_combined = np.stack((labels, snr_list), axis=1)

    x_train, x_test, labels_snr_train, labels_snr_test = train_test_split(x_combined,
                                                                          labels_snr_combined,
                                                                          test_size=test_size)

    y_train = labels_snr_train[:, 0]
    snr_train = labels_snr_train[:, 1]

    y_test = labels_snr_test[:, 0]
    snr_test = labels_snr_test[:, 1]

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
        snr_train,
        x_test_augmented,
        x_test_painted,
        y_test,
        snr_test,
        label_names,
    )
