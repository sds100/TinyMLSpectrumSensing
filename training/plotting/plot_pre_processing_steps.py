import numpy.typing as npt
from matplotlib import pyplot as plt

import spectrum_painting_data as sp_data
import spectrum_painting_training as sp_training
from training.spectrum_painting import augment_spectrogram, downsample_spectrogram, paint_spectrogram

classes = ["ZBW"]
snr = 30

spectrum_painting_options = sp_training.SpectrumPaintingTrainingOptions(
    downsample_resolution=64,
    k=3,
    l=16,
    d=4
)

high_freq_resolution_spec = sp_data.load_spectrograms(data_dir="../data/numpy",
                                                      classes=classes,
                                                      snr_list=[snr],
                                                      windows_per_spectrogram=64,
                                                      window_length=1024,
                                                      nfft=64).get(snr)[0].values

high_time_resolution_spec = sp_data.load_spectrograms(data_dir="../data/numpy",
                                                      classes=classes,
                                                      snr_list=[snr],
                                                      windows_per_spectrogram=1024,
                                                      window_length=64,
                                                      nfft=64).get(snr)[0].values

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), constrained_layout=True)


def plot_spectrogram(spectrogram: npt.NDArray, index: int, name: str):
    plt.subplot(1, 2, index)
    plt.pcolormesh(spectrogram, cmap='viridis')
    plt.title(name)
    plt.xticks((0, spectrogram.shape[1]))
    plt.xticks()


plot_spectrogram(high_freq_resolution_spec, 1, "64 windows x 1024 samples")
plot_spectrogram(high_time_resolution_spec, 2, "1024 windows x 64 samples")

fig.supxlabel("Frequency bins")
fig.supylabel("Time bins")
plt.savefig("../output/figures/stft-resolution-comparison.png")
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), constrained_layout=True)

plot_spectrogram(downsample_spectrogram(high_freq_resolution_spec, 64), 1, "64 windows x 1024 samples")
plot_spectrogram(downsample_spectrogram(high_time_resolution_spec, 64), 2, "1024 windows x 64 samples")

fig.supxlabel("Frequency bins")
fig.supylabel("Time bins")
plt.show()

spec = sp_data.load_spectrograms(data_dir="../data/numpy",
                                 classes=classes,
                                 snr_list=[snr],
                                 windows_per_spectrogram=256,
                                 window_length=256,
                                 nfft=64).get(snr)[0].values

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(4, 4), constrained_layout=True)


def plot_spectrogram(spectrogram: npt.NDArray, index: int, name: str):
    plt.subplot(1, 3, index)
    plt.pcolormesh(spectrogram, cmap='viridis')
    plt.title(name)
    plt.xticks((0, spectrogram.shape[1]))
    plt.xticks()


plot_spectrogram(spec.clip(max=0.04), index=1, name="Raw")

downsampled = downsample_spectrogram(spectrogram=spec, resolution=64)

augmented = augment_spectrogram(spectrogram=downsampled, k=3, l=16, d=4)
plot_spectrogram(augmented, index=2, name="Augmented")

painted = paint_spectrogram(downsampled, augmented)
plot_spectrogram(painted, index=3, name="Painted")

# fig.tight_layout()
fig.supxlabel("Frequency bins")
fig.supylabel("Time bins")
plt.savefig("../output/figures/raw-augmented-painted.png")
plt.show()

plt.figure(figsize=(1, 4))
plt.pcolormesh(spec, cmap='viridis')
plt.axis("off")
plt.yticks(None)
plt.savefig("../output/figures/spectrogram.png")
plt.show()

plt.figure(figsize=(1, 4))
plt.pcolormesh(downsampled, cmap='viridis')
plt.axis("off")
plt.yticks(None)
plt.savefig("../output/figures/spectrogram-downsampled.png")
plt.show()

plt.figure(figsize=(1, 4))
plt.pcolormesh(augmented, cmap='viridis')
plt.axis("off")
plt.yticks(None)
plt.savefig("../output/figures/augmented.png")
plt.show()

plt.figure(figsize=(1, 4))
plt.pcolormesh(painted, cmap='viridis')
plt.axis("off")
plt.yticks(None)
plt.savefig("../output/figures/painted.png")
plt.show()
