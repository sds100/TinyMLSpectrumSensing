import matplotlib.pyplot as plt
import numpy as np
import serial

from training.spectrogram import move_front_half_to_end

# Configure the serial port to which the Arduino is connected
ser = serial.Serial('/dev/cu.usbmodem2101', 115200, timeout=3000)

NUM_WINDOWS = 50


def read_data():
    spectrogram_data = []
    for _ in range(NUM_WINDOWS):
        real_line = ser.readline().decode('utf-8').strip()
        # imag_line = ser.readline().decode('utf-8').strip()
        magnitudes = list(map(float, real_line.split(',')))
        # imags = list(map(float, imag_line.split(',')))

        # magnitudes = []
        # 
        # for (r, i) in list(zip(reals, imags)):
        #     magnitudes.append(np.abs(complex(r, i)))

        spectrogram_data.append(magnitudes)

    spectrogram_data = move_front_half_to_end(np.asarray(spectrogram_data).T)

    return spectrogram_data


def plot_spectrogram(data):
    # data = move_front_half_to_end(data.T)
    plt.imshow(data, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time Window')
    plt.ylabel('Frequency Bin')
    plt.title('Spectrogram Arduino')
    plt.show()


# Read the data from the serial port
data = read_data()
duration = int(ser.readline().strip())
print(f"Duration = {duration} ms")
# data = move_front_half_to_end(data.T)

print(data.shape)
# Plot the spectrogram
plot_spectrogram(data)
