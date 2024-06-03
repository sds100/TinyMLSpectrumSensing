from typing import List

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from scipy.fft import fft

from training.spectrogram import move_front_half_to_end

file = "../../training/data/numpy/SNR30_ZB.npy"
data: npt.NDArray[np.complex128] = np.load(file)

NUM_WINDOWS = 100
SAMPLES = 256
data = data[:SAMPLES * NUM_WINDOWS]

fs = 88000000

data = data * 1000

print(data[0:64])

real_list: List[np.float16] = []
imag_list: List[np.float16] = []

for n in data:
    real_list.append(np.float16(n.real))
    imag_list.append(np.float16(n.imag))

spectrogram_data = []

for w in range(NUM_WINDOWS):
    start = w * SAMPLES
    end = start + SAMPLES
    old_fft_input = data[start:end]
    # result = fft(, norm="backward")
    
    fft_input: [complex] = []

    for i in range(SAMPLES):
        fft_input.append(complex(real_list[(w * SAMPLES) + i], imag_list[(w * SAMPLES) + i]))
        
    fft_input = np.asarray(fft_input)

    result = fft(fft_input, norm="backward", n=SAMPLES)
    result = np.abs(result)

    spectrogram_data.append(result)

spectrogram_data = move_front_half_to_end(np.asarray(spectrogram_data).T)

# f, t, Zxx = signal.stft(x=data, fs=fs, return_onesided=False, window="hamming", nperseg=64, noverlap=0)
# Zxx = move_front_half_to_end(Zxx)
# f = move_front_half_to_end(f)
# 
# print(Zxx.shape)
# # Zxx = fft(data)
# Zxx_abs = (np.abs(Zxx) ** 2)

plt.imshow(spectrogram_data, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Magnitude')
plt.xlabel('Time Window')
plt.ylabel('Frequency Bin')
plt.title('Spectrogram Python')
plt.show()

with open("data.h", "w") as f:
    f.write("const float real[] = {\n")

    for (i, n) in enumerate(real_list):
        f.write(f"    {np.format_float_positional(n)}")

        if i < len(real_list) - 1:
            f.write(",\n")

    f.write("\n};\n\n")

    f.write("const float imag[] = {\n")

    for (i, n) in enumerate(imag_list):
        f.write(f"    {np.format_float_positional(n)}")

        if i < len(imag_list) - 1:
            f.write(",\n")

    f.write("\n};")
