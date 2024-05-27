from typing import List

import numpy as np
import numpy.typing as npt
import scipy.fft
import scipy.io as sio
from scipy import signal
from scipy.fft import fft, ifft

file = "../../training/data/matlab/SNR30_ZBW.mat"
data: npt.NDArray[np.complex128] = sio.loadmat(file)["WaveformOut"].squeeze()

data = data[:4]

real_list: List[float] = []
imag_list: List[float] = []

for n in data:
    real_list.append(n.real)
    imag_list.append(n.imag)

with open("data.h", "w") as f:
    f.write("float real[] = {\n")

    for (i, n) in enumerate(real_list):
        f.write(f"    {n}")

        if i < len(real_list) - 1:
            f.write(",\n")

    f.write("\n};")

    f.write("float imag[] = {\n")

    for (i, n) in enumerate(imag_list):
        f.write(f"    {n}")

        if i < len(real_list) - 1:
            f.write(",\n")

    f.write("\n};")

fs = 20000000
# f, t, Zxx = signal.stft(x=data[:4], fs=fs, return_onesided=False, window="hann")
Zxx = fft(data[:4])

print(np.abs(Zxx))