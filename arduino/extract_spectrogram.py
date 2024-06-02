from typing import List

import numpy as np
import numpy.typing as npt
import scipy.io as sio
from matplotlib import pyplot as plt
from scipy import signal
from scipy.signal import windows

file = "../../training/data/matlab/SNR30_ZBW.mat"
data: npt.NDArray[np.complex128] = sio.loadmat(file)["WaveformOut"].squeeze()

NUM_WINDOWS = 10
SAMPLES = 64
data = data[:SAMPLES*NUM_WINDOWS]

fs = 20000000
f, t, Zxx = signal.stft(x=data, fs=fs, return_onesided=False, window="hann", nperseg=64)
# Zxx = fft(data)
Zxx_abs = (np.abs(Zxx) ** 2)


plt.imshow(Zxx_abs, aspect='auto', origin='lower', cmap='viridis', extent=[0, NUM_WINDOWS, 0, SAMPLES / 2])
plt.colorbar(label='Magnitude')
plt.xlabel('Time Window')
plt.ylabel('Frequency Bin')
plt.title('Spectrogram')
plt.show()

real_list: List[float] = []
imag_list: List[float] = []

for n in data:
    real_list.append(n.real)
    imag_list.append(n.imag)

with open("data.h", "w") as f:
    f.write("float real[] = {\n")

    for (i, n) in enumerate(real_list):
        f.write(f"    {np.format_float_positional(n)}")

        if i < len(real_list) - 1:
            f.write(",\n")

    f.write("\n};\n\n")

    f.write("float imag[] = {\n")

    for (i, n) in enumerate(imag_list):
        f.write(f"    {np.format_float_positional(n)}")

        if i < len(real_list) - 1:
            f.write(",\n")

    f.write("\n};")
