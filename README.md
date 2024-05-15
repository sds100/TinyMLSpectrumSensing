# Spectrum sensing on Arduino with CNN

This project is split into two for the training with Python in the `training` folder and doing inference on the Arduino in the `arduino` folder.

## Setup

## Python

1. Install Python requirements with `pip install -r requirements.txt`
2. For plotting TensorFlow models you also need the `graphviz` library to be installed on your system, which can be done with `brew install graphviz` or `sudo apt install graphviz`.

## Arduino

1. Install the Arduino IDE.
2. Follow their instructions on getting started with TensorFLow Lite Micro library. https://docs.arduino.cc/tutorials/nano-33-ble-sense/get-started-with-machine-learning/
3. Open the spectrum-painting.ino file in the Arduino IDE.
