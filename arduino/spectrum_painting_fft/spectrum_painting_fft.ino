#include <arduinoFFT.h>
#include "data.h"
#include "kiss_fft.h"

const uint16_t SAMPLES = 256;
const uint16_t NFFT = 64;
const float SAMPLING_FREQUENCY = 88000000;
const int NUM_WINDOWS = 400;

unsigned int sampling_period_us;
unsigned long microseconds;

kiss_fft_cfg cfg;
kiss_fft_cpx in[SAMPLES];
kiss_fft_cpx out[NFFT];

// Example I/Q data (replace this with your actual data)
// For the sake of this example, we'll just simulate some data.
// float iqData[NUM_WINDOWS][SAMPLES][2];

float spectrogram[NFFT * NUM_WINDOWS];

void setup() {
  Serial.begin(115200);
  while (!Serial)
    ;

  // sampling_period_us = round(1000000 * (1.0 / SAMPLING_FREQUENCY));

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
  unsigned long timeBegin = millis();

  kiss_fft_cfg cfg = kiss_fft_alloc(NFFT, false, 0, 0);

  // Collect data and perform FFT for each window
  for (int w = 0; w < NUM_WINDOWS; w++) {
    for (int i = 0; i < SAMPLES; i++) {
      // vReal[i] = real[w + i];  // Real part (I)
      // vImag[i] = imag[w + i];  // Imaginary part (Q)
      int memIndex = (w * SAMPLES) + i;

      // Serial.print("Read index " + memIndex);
      in[i].r = pgm_read_float(real + memIndex);
      in[i].i = pgm_read_float(imag + memIndex);
    }
    
    kiss_fft(cfg, in, out);

    // ArduinoFFT<float> FFT = ArduinoFFT<float>(vReal, vImag, SAMPLES, SAMPLING_FREQUENCY, true); /* Create FFT object */

    // // Perform FFT
    // FFT.windowing(FFTWindow::Hamming, FFTDirection::Forward);  // Apply window function (Hamming)
    // FFT.compute(FFTDirection::Forward);                        // Compute FFT
    // FFT.complexToMagnitude();                                  // Compute magnitudes

    // Send results over serial
    for (int i = 0; i < NFFT; i++) {  // Only output the first half of the FFT results (real frequency components)
      float magnitude = sqrt(out[i].r * out[i].r + out[i].i * out[i].i);

      spectrogram[(w * NFFT) + i] = magnitude;
    }
  }

  unsigned long timeEnd = millis(); 
  unsigned long duration = timeEnd - timeBegin;

  for (int w = 0; w < NUM_WINDOWS; w++){
    for (int i = 0; i < NFFT; i++){
      Serial.print(spectrogram[(w * NFFT) + i]);

      if (i < (NFFT) - 1) {
        Serial.print(",");
      }
    }

    Serial.println();
  }


  Serial.println(duration);

  while (true)
    ;
}
