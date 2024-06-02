#include <arduinoFFT.h>
#include "data.h"

const uint16_t SAMPLES = 64;
const float SAMPLING_FREQUENCY = 20000000;
const int NUM_WINDOWS = 10;

unsigned int sampling_period_us;
unsigned long microseconds;

// float vReal[SAMPLES]; // Real part
// float vImag[SAMPLES]; // Imaginary part (all zeros for I/Q data)

ArduinoFFT<float> FFT = ArduinoFFT<float>(real, imag, SAMPLES, SAMPLING_FREQUENCY); /* Create FFT object */

// Example I/Q data (replace this with your actual data)
// For the sake of this example, we'll just simulate some data.
// float iqData[NUM_WINDOWS][SAMPLES][2];

void setup() {
  Serial.begin(115200);
  while (!Serial);

  sampling_period_us = round(1000000 * (1.0 / SAMPLING_FREQUENCY));

  // Simulate I/Q data (replace with actual I/Q data collection)
  // for (int w = 0; w < NUM_WINDOWS; w++) {
  //   for (int i = 0; i < SAMPLES; i++) {
  //     iqData[w][i][0] = sin(2 * PI * i / SAMPLES) * cos(2 * PI * w / NUM_WINDOWS); // I component
  //     iqData[w][i][1] = sin(2 * PI * i / SAMPLES) * sin(2 * PI * w / NUM_WINDOWS); // Q component
  //   }
  // }
}

void loop() {
  delay(2000);
    // Collect data and perform FFT for each window
  for (int w = 0; w < NUM_WINDOWS; w++) {
    // for (int i = 0; i < SAMPLES; i++) {
    //   vReal[i] = iqData[w][i][0];  // Real part (I)
    //   vImag[i] = iqData[w][i][1];  // Imaginary part (Q)
    // }

    // Perform FFT
    FFT.windowing(real, SAMPLES, FFTWindow::Hann, FFTDirection::Forward);  // Apply window function (Hamming)
    FFT.compute(real, imag, SAMPLES, FFTDirection::Forward);                  // Compute FFT
    FFT.complexToMagnitude(real, imag, SAMPLES);                    // Compute magnitudes

    // Send results over serial
    for (int i = 0; i < (SAMPLES / 2); i++) {  // Only output the first half of the FFT results (real frequency components)
      Serial.print(real[i], 8);               // Adjust precision as needed
      if (i < (SAMPLES) - 1) {
        Serial.print(",");
      }
    }
    Serial.println();
  }

  while (true);
}
