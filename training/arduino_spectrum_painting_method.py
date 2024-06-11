from typing import List

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from scipy.fft import fft

file = "data/numpy/SNR10_ZBW.npy"
data: npt.NDArray[np.complex64] = np.load(file)

indices = np.arange(0, len(data), step=4)
data = data[indices]

# This file follows the same method as the Arduino for spectrum
# painting, so it is easier to verify what it is creating
# is correct.

NUM_WINDOWS = 256
SAMPLES = 256
NFFT = 256
TARGET_RESOLUTION = 64

data_offset = 0
data = data[data_offset:data_offset + (SAMPLES * NUM_WINDOWS)]

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

spectrogram_data: npt.NDArray[float] = np.zeros(shape=(NUM_WINDOWS, NFFT))

for w in range(NUM_WINDOWS):
    time_start = w * SAMPLES
    end = time_start + SAMPLES

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

    for i in range(len(result)):
        spectrogram_data[w][i] = result[i]

time_scale_factor = NUM_WINDOWS // TARGET_RESOLUTION
freq_scale_factor = NFFT // TARGET_RESOLUTION

downsampled: npt.NDArray[float] = np.zeros(shape=TARGET_RESOLUTION * TARGET_RESOLUTION)

for i in range(TARGET_RESOLUTION):
    time_start = i * time_scale_factor

    for j in range(TARGET_RESOLUTION):
        freq_start = j * freq_scale_factor
        time_sum = 0

        for k in range(time_scale_factor):
            freq_sum = 0

            for m in range(freq_scale_factor):
                freq_sum += spectrogram_data[time_start + k][freq_start + m]

            time_sum += freq_sum

        column_scaled = time_sum / time_scale_factor

        downsampled[i * TARGET_RESOLUTION + j] = column_scaled

plt.imshow(downsampled.reshape((64, 64)))
plt.colorbar(label='Magnitude')
plt.xlabel('Frequency Window')
plt.ylabel('Time Window')
plt.title('Downsampled spectrogram Python')
plt.show()

freq_bins = TARGET_RESOLUTION
time_bins = TARGET_RESOLUTION
K = 3
L = 16
D = 4

augmented_freq_bins = ((freq_bins - L) // D) + 1
output_length = augmented_freq_bins * time_bins

augmented: npt.NDArray[float] = np.zeros(shape=output_length)

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
        value: float = downsampled[(t * freq_bins) + f] - input_mean
        augmented[(t * augmented_freq_bins) + f_augmented] = max(0, value)

        f_augmented += 1
        f += D


def digitize(spectrogram: npt.NDArray[float]) -> npt.NDArray[np.uint8]:
    digitized_spectrogram: npt.NDArray[np.uint8] = np.zeros(shape=time_bins * augmented_freq_bins)
    max_value: float = 0

    for t in range(time_bins):
        for f in range(augmented_freq_bins):
            value = spectrogram[(t * augmented_freq_bins) + f]
            max_value = max(max_value, value)

    scale_factor = 255 / max_value

    if max_value == 0:
        return digitized_spectrogram

    for t in range(time_bins):
        for f in range(augmented_freq_bins):
            index = (t * augmented_freq_bins) + f
            value = spectrogram[index] * scale_factor

            if value < 0:
                value = 0

            digitized_spectrogram[index] = value

    return digitized_spectrogram


augmented_digitized = digitize(augmented)

plt.imshow(np.asarray(augmented_digitized).reshape((TARGET_RESOLUTION, augmented_freq_bins)))
plt.colorbar(label='Magnitude')
plt.xlabel('Frequency Window')
plt.ylabel('Time Window')
plt.title('Spectrogram Augmented Python')
plt.show()

painted: npt.NDArray[float] = np.zeros(shape=output_length)

for t in range(time_bins):
    mean_time_original = 0

    for f in range(freq_bins):
        mean_time_original += downsampled[(t * freq_bins) + f]

    mean_time_original /= freq_bins

    for f in range(augmented_freq_bins):
        augmented_value: float = augmented[(t * augmented_freq_bins) + f]
        painted[(t * augmented_freq_bins) + f] = max(0, augmented_value - mean_time_original)

painted_digitized = digitize(painted)

plt.imshow(np.asarray(painted_digitized).reshape((TARGET_RESOLUTION, augmented_freq_bins)))
plt.colorbar(label='Magnitude')
plt.xlabel('Frequency Window')
plt.ylabel('Time Window')
plt.title('Spectrogram Painted Python')
plt.show()
