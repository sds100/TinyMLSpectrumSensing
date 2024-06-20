import numpy.typing as npt
from matplotlib import pyplot as plt

import spectrum_painting_data as sp_data
from training.spectrum_painting import augment_spectrogram, downsample_spectrogram

snr_list = [-100, -15, -10, -5, 0, 5, 10, 15, 20, 30]

spectrogram = sp_data.load_spectrograms(data_dir="../data/numpy",
                                        classes=["ZBW"],
                                        snr_list=snr_list,
                                        windows_per_spectrogram=256,
                                        window_length=256,
                                        nfft=64)

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(5, 6), constrained_layout=True)


def plot_spectrogram(spectrogram: npt.NDArray, index: int, name: str):
    plt.subplot(2, 5, index)
    plt.pcolormesh(spectrogram, cmap='viridis')
    plt.title(name)
    plt.xticks((0, spectrogram.shape[1]))
    plt.xticks()


for i, snr in enumerate(snr_list):
    spec = spectrogram[snr][0].values

    downsampled = downsample_spectrogram(spectrogram=spec, resolution=64)
    augmented = augment_spectrogram(spectrogram=downsampled, k=3, l=16, d=4)

    plot_spectrogram(augmented, i + 1, f"SNR {snr}")
    
plt.savefig("../output/figures/different-snr-augmented.png")
plt.show()
