import numpy as np
import numpy.typing as npt
from skimage.transform import downscale_local_mean

from spectrogram import Spectrogram


def take_frequencies(spec: Spectrogram, start: int, end: int) -> Spectrogram:
    # for spectrum painting to work best, the WiFi signal must
    # fill the spectrogram so only take the frequencies for the
    # WiFi signal
    spec.f = spec.f[start:end]
    spec.values = spec.values[start:end]
    return spec


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
    downsampled_spec_values = np.flip(downsampled_spec_values, axis=1)

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
    spectrogram_copy = np.copy(spectrogram).T
    (M, N) = spectrogram_copy.shape

    augmented_width = (N - l) // d + 1
    augmented_spectrogram: np.ndarray = np.zeros(shape=(M, augmented_width), dtype=float)

    m = np.mean(spectrogram_copy)

    for i in range(1, M):
        j_augmented = 0
        j = 0
        while j <= N - l:
            # For the current row i, get a window of size L.
            window = spectrogram_copy[i][j:(j + l)]
            # Get the top K elements by sorting and taking last K elements.
            top_k = np.sort(window)[len(window) - k:]
            mean_top_k = np.mean(top_k)
            spectrogram_copy[i][j] = mean_top_k
            augmented_spectrogram[i][j_augmented] = spectrogram_copy[i][j] - m

            j_augmented += 1
            j += d

    return augmented_spectrogram.clip(0, 1)


def paint_spectrogram(original: npt.NDArray, augmented: npt.NDArray) -> npt.NDArray:
    painted_spectrogram = np.zeros(shape=augmented.shape)
    mean_original_rows = np.mean(original, axis=0)

    for row in range(len(augmented)):
        painted_spectrogram[row] = augmented[row] - mean_original_rows[row]

    # clip the values so they are all positive
    return painted_spectrogram.clip(min=0)


def digitize_spectrogram(spectrogram: npt.NDArray[np.float32], color_depth: int) -> npt.NDArray[np.uint8]:
    """
    Digitize the spectrogram from a range of floating point numbers to discrete integers.

    :param spectrogram: The spectrogram to digitize.
    :param color_depth: How many discrete values one "pixel" in the spectrogram can have. Since
                        this returns a byte array then the max value is 256.
    """
    max_value: float = spectrogram.max()
    scaled_spectrogram: npt.NDArray[np.float32]

    if max_value == 0:
        scaled_spectrogram = spectrogram
    else:
        scale: float = color_depth / max_value
        spectrogram = spectrogram.clip(min=0)
        scaled_spectrogram = spectrogram * scale

    # Must be UNSIGNED int so that the bins are
    # monotonically increasing.
    bins = np.arange(color_depth, dtype=np.uint8)
    bin_indices = np.digitize(scaled_spectrogram, bins)
    return bins[bin_indices - 1]
