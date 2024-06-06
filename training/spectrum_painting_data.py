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
                      sample_rate: int,
                      count: int,
                      window_length: int) -> Dict[str, List[Spectrogram]]:
    """
    Read the time-domain data and convert it to spectrograms.

    :param data_dir: The directory where the data files are stored. The files must be named with
                    the following pattern: SNR_{snr}_{class}.npy where snr is the signal-to-noise ratio
                    and class is which types of signals are present - e.g Z for Zigbee, ZW for Zigbee and WiFi.
                    The .npy files should contain an array of np.complex128 numbers.
    :param classes: The classes to load.
    :param snr_list: The signal-to-noise ratios to load.
    :param sample_rate: The sample rate of the data.
    :param count: How many lines in the input files to read from.
    :param window_length: The length of the window in the Short-Time Fourier Transform. This corresponds to the resolution
                            in the frequency axis.

    :return: A dictionary that maps each class to a list of spectrograms with
             different signal-to-noise ratios.
    """
    spectrograms: Dict[str, List[Spectrogram]] = {}

    for snr in snr_list:
        spectrograms[snr] = []

        for label in classes:
            data = np.load(f"{data_dir}/SNR{snr}_{label}.npy")

            spectrogram = create_spectrogram(data[:count], sample_rate, label, window_length)
            spectrograms[snr].append(spectrogram)

            del data

    return spectrograms
