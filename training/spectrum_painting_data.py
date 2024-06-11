from typing import Dict, List

import numpy as np
import numpy.typing as npt
import scipy.io as sio

from training.spectrogram import Spectrogram, create_spectrogram


def load_data_from_matlab(file: str) -> npt.NDArray[np.complex128]:
    """
    Load the list of complex numbers from Matlab files.
    """
    # each complex number is in its own row and so is put in
    # its own array. 'squeeze' flattens the array.
    return sio.loadmat(file)["WaveformOut"].squeeze()


def convert_matlab_to_numpy(matlab_dir: str,
                            numpy_dir: str,
                            classes: List[str],
                            snr_list: List[int]):
    """
    :param matlab_dir: The directory where the data files are stored. The files must be named with
                    the following pattern: SNR_{snr}_{class}.mat where snr is the signal-to-noise ratio
                    and class is which types of signals are present - e.g Z for Zigbee, ZW for Zigbee and WiFi.
                    The .mat files should contain one object called "WaveformOut".
    """
    for c in classes:
        for snr in snr_list:
            data = load_data_from_matlab(f"{matlab_dir}/SNR{snr}_{c}.mat")

            np.save(f"{numpy_dir}/SNR{snr}_{c}.npy", data)


def load_spectrograms(data_dir: str,
                      classes: List[str],
                      snr_list: List[int],
                      windows_per_spectrogram: int,
                      window_length: int,
                      nfft: int) -> Dict[int, List[Spectrogram]]:
    """
    Read the time-domain data and convert it to spectrograms.

    :param data_dir: The directory where the data files are stored. The files must be named with
                    the following pattern: SNR_{snr}_{class}.npy where snr is the signal-to-noise ratio
                    and class is which types of signals are present - e.g Z for Zigbee, ZW for Zigbee and WiFi.
                    The .npy files should contain an array of np.complex128 numbers.
    :param classes: The classes to load.
    :param snr_list: The signal-to-noise ratios to load.
    :param window_length: The length of each window to perform the FFT on.

    :return: A dictionary that maps each SNR to a list of Spectrograms.
    """
    spectrograms: Dict[int, List[Spectrogram]] = {}

    for snr in snr_list:
        spectrograms[snr] = []

        for label in classes:
            data = np.load(f"{data_dir}/SNR{snr}_{label}.npy")

            # Use a step size of 4 to reduce the number of I/Q samples.
            # This effectively 
            # This also has the knock-on effect of only outputting
            # frequencies in the FFT that are between +-22 MHz. This means
            # the Wi-Fi signal fills the spectrogram.
            indices = np.arange(0, len(data), step=4)
            data = data[indices]

            samples_per_spectrogram = windows_per_spectrogram * window_length
            spectrogram_count = len(data) // samples_per_spectrogram

            for i in range(spectrogram_count):
                start = i * samples_per_spectrogram
                end = start + samples_per_spectrogram
                data_slice = data[start:end]

                spectrogram = create_spectrogram(data_slice, label, windows_per_spectrogram, window_length, nfft)
                spectrograms[snr].append(spectrogram)

            del data

    return spectrograms
