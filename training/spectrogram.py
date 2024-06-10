from dataclasses import dataclass
from typing import List

import numpy as np
import numpy.typing as npt
from scipy.fft import fft


@dataclass
class Spectrogram:
    """
    This stores the information required to interpret a spectrogram.

    values: is a 2D numpy array for the magnitudes for each frequency f at time t.
    snr: The signal-to-noise ratio of the spectrogram.
    """
    values: npt.NDArray[np.float32]
    label: str


def move_front_half_to_end(array: npt.NDArray) -> npt.NDArray:
    """
    Move the front half of an array to the end
    """
    n = len(array)
    return np.concatenate((array[n // 2:], array[:n // 2]))


def create_spectrogram(x: npt.NDArray[np.complex64], fs: int, label: str, window_length: int) -> Spectrogram:
    """
    Create a spectrogram of a signal using the same simplified STFT method that will be done on the Arduino.
    This is about 2x faster than calling the scipy STFT function.
    :param x: The signal
    :param fs: The sample rate in Hz.
    :return: A Spectrogram instance that contains the frequencies, time and values.
    """
    samples = 256
    windows = len(x) // 256

    spectrogram_values = np.empty(shape=(windows, window_length))

    for w in range(windows):
        start = w * samples
        end = start + samples

        fft_input = x[start:end]

        result = fft(fft_input, norm="backward", n=window_length)

        result = np.abs(result)

        middle = len(result) // 2

        # For some reason our spectrogram has the negative frequencies in
        # the *last* half of the data so this method moves to the front
        # This can help with plotting because to use 'nearest' shading requires the
        # data to be monotonically increasing/decreasing in value.
        result = np.concatenate((result[middle:], result[:middle]))

        spectrogram_values[w] = result

    return Spectrogram(values=spectrogram_values, label=label)


def split_spectrogram(spectrogram: npt.NDArray, duration: int) -> List[npt.NDArray]:
    """
    Split up a spectrogram along the time axis into a bunch of smaller spectrograms.
    :param spectrogram: The spectrogram to split
    :param duration: The length of each sub-spectrogram
    :return: A list of sub spectrograms.
    """
    sub_spectrograms: List[npt.NDArray] = []

    start_index: int = 0
    end_index: int = duration

    while end_index < spectrogram.shape[0]:
        values_slice = spectrogram[start_index:end_index]

        sub_spectrograms.append(np.asarray(values_slice))

        start_index += duration
        end_index += duration

    return sub_spectrograms
