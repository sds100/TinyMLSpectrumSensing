import matplotlib.pyplot as plt
import serial

# Configure the serial port to which the Arduino is connected
ser = serial.Serial('/dev/cu.usbmodem2101', 115200, timeout=3000)

# The number of rows being outputted by the arduino
OUTPUT_LENGTH = 64


def read_data():
    spectrogram_data = []
    for _ in range(OUTPUT_LENGTH):
        real_line = ser.readline().decode('utf-8').strip()
        magnitudes = list(map(float, real_line.split(',')))

        spectrogram_data.append(magnitudes)

    return spectrogram_data


def plot_spectrogram(data):
    # data = move_front_half_to_end(data.T)
    plt.imshow(data)
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

# Plot the spectrogram
plot_spectrogram(data)
