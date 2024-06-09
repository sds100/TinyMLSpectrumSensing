from typing import List, TextIO

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from scipy.fft import fft

file = "../training/data/numpy/SNR30_ZBW.npy"
data: npt.NDArray[np.complex64] = np.load(file)

NUM_WINDOWS = 256
SAMPLES = 256
NFFT = 64
data = data[:SAMPLES * NUM_WINDOWS]

fs = 88000000

max_value = np.max(data).real

data_scale_factor: float = 128 / max_value
data = data * data_scale_factor

real_list: List[str] = []
imag_list: List[str] = []


def format_num(n) -> str:
    # return np.format_float_positional(np.float16(n))
    return str(np.int8(n))


for n in data:
    real_list.append(format_num(n.real))
    imag_list.append(format_num(n.imag))

spectrogram_data = []

for w in range(NUM_WINDOWS):
    start = w * SAMPLES
    end = start + SAMPLES

    fft_input: [complex] = []

    for i in range(SAMPLES):
        number_index = (w * SAMPLES) + i

        real = np.int8(real_list[number_index])
        imag = np.int8(imag_list[number_index])

        fft_input.append(complex(real, imag))

    fft_input = np.asarray(fft_input)

    result = fft(fft_input, norm="backward", n=NFFT)

    result = np.abs(result)

    middle = len(result) // 2

    result = np.concatenate((result[middle:], result[:middle]))

    spectrogram_data.append(list(result))

target_resolution = 64
scale_factor = NUM_WINDOWS // target_resolution

downsampled: npt.NDArray[float] = np.empty(shape=target_resolution * target_resolution)

for i in range(target_resolution):
    start = i * scale_factor

    for j in range(NFFT):
        col_sum = 0

        for k in range(scale_factor):
            col_sum = col_sum + spectrogram_data[start + k][j]

        column_scaled = col_sum / scale_factor

        downsampled[i * target_resolution + j] = column_scaled

# f, t, Zxx = signal.stft(x=data, fs=fs, return_onesided=False, window="hamming", nperseg=64, noverlap=0)
# Zxx = move_front_half_to_end(Zxx)
# f = move_front_half_to_end(f)
# 
# print(Zxx.shape)
# # Zxx = fft(data)
# Zxx_abs = (np.abs(Zxx) ** 2)

plt.imshow(np.asarray(downsampled).reshape((64, 64)), aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(label='Magnitude')
plt.xlabel('Time Window')
plt.ylabel('Frequency Bin')
plt.title('Spectrogram Python')
plt.show()

freq_bins = NFFT
time_bins = target_resolution
K = 3
L = 16
D = 4

augmented_freq_bins = ((freq_bins - L) // D) + 1
output_length = augmented_freq_bins * time_bins

augmented: npt.NDArray = np.zeros(shape=output_length)

input_mean = np.mean(downsampled)

for t in range(time_bins):
    f_augmented = 0
    f = 0

    while f <= (freq_bins - L):
        start_of_window = (t * freq_bins) + f
        window = downsampled[start_of_window:(start_of_window + L)]

        window = np.sort(window)
        mean_top_k = 0

        for i in range(K):
            mean_top_k += window[len(window) - i - 1]

        mean_top_k /= K

        downsampled[(t * freq_bins) + f] = mean_top_k
        augmented[(t * augmented_freq_bins) + f_augmented] = downsampled[(t * freq_bins) + f] - input_mean

        f_augmented += 1
        f += D

plt.imshow(np.asarray(augmented).reshape((NFFT, augmented_freq_bins)))
plt.colorbar(label='Magnitude')
plt.xlabel('Time Window')
plt.ylabel('Frequency Bin')
plt.title('Spectrogram Augmented Python')
plt.show()


def write_variable(x: List, f: TextIO, name: str, type: str):
    f.write(f"const static {type} {name}[] PROGMEM = " + "{\n")

    for (i, n) in enumerate(x):
        f.write(f"    {n}")

        if i < len(x) - 1:
            f.write(",\n")

    f.write("\n};\n\n")


with open("spectrum_painting/data.h", "w") as f:
    f.write("#include <avr/pgmspace.h>\n")

    write_variable(real_list, f, "real", "int8_t")
    write_variable(imag_list, f, "imag", "int8_t")
