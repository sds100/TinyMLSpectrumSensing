#include <arduinoFFT.h>
#include "data.h"

const uint16_t samples = 4;  //This value MUST ALWAYS be a power of 2
const float samplingFrequency = 20000000;

ArduinoFFT<float> FFT = ArduinoFFT<float>(real, imag, samples, samplingFrequency); /* Create FFT object */

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(4000);
  // wait for serial initialization so printing in setup works.
  while (!Serial)
    ;
}

void loop() {
  // Get samples
  FFT.windowing(FFTWindow::Hann, FFTDirection::Forward); /* Weigh data */
  FFT.compute(FFTDirection::Forward);                    /* Compute FFT */
  FFT.complexToMagnitude();                                 /* Compute magnitudes */
  float x = FFT.majorPeak();
  // Rest of the code
  if (!std::isnan(x)) {
    Serial.println(x);
  }
  while(1);
}