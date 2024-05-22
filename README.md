# Spectrum sensing on Arduino with a Convolutional Neural Network

This project is split into two for the training with Python in the `training` folder and doing inference on the Arduino in the `arduino` folder.

## Setup

### MATLAB (optional)

To generate new data you need to use MATLAB 2022b or older. In 2023 they changed the API for generating Zigbee signals and this code is not migrated.

### GPU/CUDA (optional)

Uncomment the required Tensorflow packages depending on your system in `requirements.txt`.

See the supported dependencies table for TensorFlow here https://www.tensorflow.org/install/source#gpu.
Make sure you have compatible versions for Python, TensorFlow, C Compiler, cuDNN and CUDA.

### Python

1. Create Python 3.10 virtual environment.
2. **Dependencies**:
    1. Install Python requirements with `pip install -r requirements.txt`. If you want to run the TensorFlow *Lite* model on your PC then it must be Linux because there are no wheels available for macOS and Windows. The library `tflite-runtime` is commented out in the requirements.txt for this reason. The full TensorFlow model still works.
    2. For plotting TensorFlow models you also need the `graphviz` library to be installed on your system, which can be done with `brew install graphviz` or `sudo apt install graphviz`.
    3. `xxd` command must exist on your system so models and testing images can be converted to hexdumps in C files for the Arduino.
3. The data for training should be put in `training/data/csv`.

### Arduino

1. Install the Arduino IDE.
2. Follow their instructions on getting started with TensorFLow Lite Micro library. https://docs.arduino.cc/tutorials/nano-33-ble-sense/get-started-with-machine-learning/
3. Open the `spectrum-painting.ino` file in the Arduino IDE.

## Useful links

- Great tutorial series on how to do TinyML on the Arduino Nano BLE with TensorFlow Lite Micro https://www.youtube.com/watch?v=BzzqYNYOcWc.
- Visualize any type of ML model with https://netron.app.