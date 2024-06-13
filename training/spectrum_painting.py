import numpy as np
import numpy.typing as npt
from skimage.transform import downscale_local_mean

from spectrogram import Spectrogram


def take_frequencies(spec: Spectrogram, start: int, end: int) -> Spectrogram:
    return Spectrogram(values=spec.values.T[start:end].T, label=spec.label)


def downsample_spectrogram(spectrogram: npt.NDArray, resolution: int) -> npt.NDArray:
    """
    Downsample a spectrogram to a target N x N resolution.

    :param spectrogram: A 2D array of a spectrogram. 
    :arg resolution: The target height/width of the image in pixels.
    :return: The downsampled spectrogram.
    """

    (height, width) = spectrogram.shape
    time_factor = width // resolution
    freq_factor = height // resolution
    downsampled_spec_values = downscale_local_mean(spectrogram, (freq_factor, time_factor))[:resolution, :resolution]

    # make sure the spectrogram is still in the correct orientation
    # downsampled_spec_values = np.flip(downsampled_spec_values, axis=0)

    assert downsampled_spec_values.shape == (resolution, resolution)
    return downsampled_spec_values


def augment_spectrogram(spectrogram: npt.NDArray, k: int, l: int, d: int) -> npt.NDArray:
    """
    Augment the Bluetooth and Zigbee signals by stretching them.

    :param spectrogram: An M x N array of signal magnitude values.
    :param k: The number of maximum values
    :param l: The sliding window size
    :param d: The step size
    """

    # copy the spectrogram so modifying the values in-place
    # does not change the argument.
    spectrogram_copy = np.copy(spectrogram)
    (time_bins, freq_bins) = spectrogram_copy.shape

    augmented_width = ((freq_bins - l) // d) + 1
    augmented_spectrogram: np.ndarray = np.zeros(shape=(time_bins, augmented_width), dtype=np.float32)

    m = np.mean(spectrogram_copy)

    for t in range(time_bins):
        f_augmented = 0
        f = 0
        while f <= freq_bins - l:
            # For the current row i, get a window of size L.
            window = spectrogram_copy[t][f:(f + l)]
            # Get the top K elements by sorting and taking last K elements.
            top_k = np.sort(window)[len(window) - k:]
            mean_top_k = np.mean(top_k)
            spectrogram_copy[t][f] = mean_top_k
            augmented_spectrogram[t][f_augmented] = spectrogram_copy[t][f] - m

            f_augmented += 1
            f += d

    return augmented_spectrogram.clip(min=0)


def paint_spectrogram(original: npt.NDArray, augmented: npt.NDArray) -> npt.NDArray:
    painted_spectrogram = np.zeros(shape=augmented.shape)
    mean_original_rows = np.mean(original, axis=1)

    for row in range(len(augmented)):
        painted_spectrogram[row] = augmented[row] - mean_original_rows[row]

    # clip the values so they are all positive
    return painted_spectrogram.clip(min=0)


def digitize_spectrogram(spectrogram: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    """
    Digitize the spectrogram from a range of floating point numbers to discrete integers.

    :param spectrogram: The spectrogram to digitize.
    """
    max_value: float = spectrogram.max()
    scaled_spectrogram: npt.NDArray = np.zeros(spectrogram.shape)

    if max_value == 0:
        return scaled_spectrogram.astype(np.uint8)
    else:
        scale: float = 255 / max_value
        spectrogram = spectrogram.clip(min=0)
        scaled_spectrogram = spectrogram * scale

    return scaled_spectrogram.astype(np.uint8)
