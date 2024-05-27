from typing import Dict

import numpy as np
import numpy.typing as npt
import scipy.io as sio

from training.spectrogram import Spectrogram, create_spectrogram


def load_spectrograms(data_dir: str,
                      snr: int,
                      sample_rate: int,
                      count: int) -> Dict[str, Spectrogram]:
    """
    Read the time-domain data and convert it to spectrograms.

    :param data_dir: The directory where the data files are stored. The files must be named with
                    the following pattern: SNR_{snr}_{class}.mat where snr is the signal-to-noise ratio
                    and class is which types of signals are present - e.g Z for Zigbee, ZW for Zigbee and WiFi.
                    The .mat files should contain one object called "WaveformOut".
    :param snr: The signal-to-noise ratio of the data.
    :param sample_rate: The sample rate of the data.
    """

    def load_data_from_matlab(file: str) -> npt.NDArray[np.complex128]:
        """
        Load the list of complex numbers from Matlab files.
        """
        # each complex number is in its own row and so is put in
        # its own array. 'squeeze' flattens the array.
        return sio.loadmat(file)["WaveformOut"].squeeze()[:count]

    # change which signal-to-noise ratio to use. The files have to be called SNR_{snr}_{class}.mat

    data: Dict[str, npt.NDArray[np.complex128]] = {
        "z": load_data_from_matlab(f"{data_dir}/SNR{snr}_Z.mat"),
        "b": load_data_from_matlab(f"{data_dir}/SNR{snr}_B.mat"),
        "w": load_data_from_matlab(f"{data_dir}/SNR{snr}_W.mat"),
        "bw": load_data_from_matlab(f"{data_dir}/SNR{snr}_BW.mat"),
        "zb": load_data_from_matlab(f"{data_dir}/SNR{snr}_ZB.mat"),
        "zw": load_data_from_matlab(f"{data_dir}/SNR{snr}_ZW.mat"),
        "zbw": load_data_from_matlab(f"{data_dir}/SNR{snr}_ZBW.mat"),
    }

    spectrograms: Dict[str, Spectrogram] = {key: create_spectrogram(frame, sample_rate) for key, frame in data.items()}

    return spectrograms
