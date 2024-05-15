from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import numpy.typing as npt
from scipy import signal


@dataclass
class Spectrogram:
    """
    This stores the information required to interpret a spectrogram.

    f are the frequencies
    t are an array of time intervals
    values is a 2D numpy array for the magnitudes for each frequency f at time t.
    """
    f: List[int]
    t: List[int]
    values: npt.NDArray[np.float32]


def move_front_half_to_end(array: npt.NDArray) -> npt.NDArray:
    """
    Move the front half of an array to the end
    """
    n = len(array)
    return np.concatenate((array[n // 2:], array[:n // 2]))


def create_spectrogram(x: Iterable, fs: int) -> Spectrogram:
    """
    Create a spectrogram of a signal.
    :param x: The signal
    :param fs: The sample rate in Hz.
    :return: A Spectrogram instance that contains the frequencies, time and values.
    """

    # Calculate Short Time Fourier Transform of the signal.
    # return_one_sided is needed because the data is complex
    f, t, Zxx = signal.stft(x=x, fs=fs, return_onesided=False)

    # A spectrogram is the absolute value of the STFT and then squared.
    Zxx_abs = (np.abs(Zxx) ** 2)

    # For some reason our spectrogram has the negative frequencies in
    # the *last* half of the data so this method moves to the front
    # This can help with plotting because to use 'nearest' shading requires the
    # data to be monotonically increasing/decreasing in value.
    return Spectrogram(move_front_half_to_end(f), t, move_front_half_to_end(Zxx_abs))


def split_spectrogram(spectrogram: Spectrogram, duration: int) -> List[Spectrogram]:
    """
    Split up a spectrogram along the time axis into a bunch of smaller spectrograms.
    :param spectrogram: The spectrogram to split
    :param duration: The length of each sub-spectrogram
    :return: A list of sub spectrograms.
    """
    sub_spectrograms: List[Spectrogram] = []

    start_index: int = 0
    end_index: int = duration

    while end_index < len(spectrogram.t):
        values_slice = [time_values[start_index:end_index] for time_values in spectrogram.values]

        sub_spectrograms.append(
            Spectrogram(spectrogram.f,
                        spectrogram.t[start_index:end_index],
                        np.array(values_slice)))

        start_index += duration
        end_index += duration

    return sub_spectrograms
