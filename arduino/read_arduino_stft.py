import serial
import numpy as np
import matplotlib.pyplot as plt

# Configure the serial port to which the Arduino is connected
ser = serial.Serial('/dev/cu.usbmodem2101', 115200, timeout=3000)  # Adjust 'COM3' to your serial port

NUM_WINDOWS = 10
SAMPLES = 64

def read_data():
    spectrogram_data = []
    for _ in range(NUM_WINDOWS):
        line = ser.readline().decode('utf-8').strip()
        magnitudes = list(map(float, line.split(',')))
        spectrogram_data.append(magnitudes)
        
    print(spectrogram_data)
    return np.array(spectrogram_data)

def plot_spectrogram(data):
    plt.imshow(data.T, aspect='auto', origin='lower', cmap='viridis', extent=[0, NUM_WINDOWS, 0, SAMPLES / 2])
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time Window')
    plt.ylabel('Frequency Bin')
    plt.title('Spectrogram')
    plt.show()

# Read the data from the serial port
data = read_data()
print(data.shape)
# Plot the spectrogram
plot_spectrogram(data)