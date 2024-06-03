#include <arduinoFFT.h>
#include "data.h"
#include "kiss_fft.h"

const uint16_t SAMPLES = 256;
const uint16_t NFFT = 64;
const float SAMPLING_FREQUENCY = 88000000;
const int NUM_WINDOWS = 512;
const int TARGET_RESOLUTION = 64;

unsigned int sampling_period_us;
unsigned long microseconds;

kiss_fft_cfg cfg;
kiss_fft_cpx in[SAMPLES];
kiss_fft_cpx out[NFFT];

// Example I/Q data (replace this with your actual data)
// For the sake of this example, we'll just simulate some data.
// float iqData[NUM_WINDOWS][SAMPLES][2];

float spectrogram[NFFT * NUM_WINDOWS];
float downsampled[NFFT * TARGET_RESOLUTION];

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
      
      // Don't need to rescale the data. Doing FFT on integers works fine.
      in[i].r = ((int8_t) pgm_read_byte(real + memIndex));
      in[i].i = ((int8_t) pgm_read_byte(imag + memIndex));
    }
    
    kiss_fft(cfg, in, out);

    // ArduinoFFT<float> FFT = ArduinoFFT<float>(vReal, vImag, SAMPLES, SAMPLING_FREQUENCY, true); /* Create FFT object */

    // // Perform FFT
    // FFT.windowing(FFTWindow::Hamming, FFTDirection::Forward);  // Apply window function (Hamming)
    // FFT.compute(FFTDirection::Forward);                        // Compute FFT
    // FFT.complexToMagnitude();                                  // Compute magnitudes

    // Send results over serial
    int middle = NFFT / 2;

    // I'm not sure why but for my training data, computing the FFT puts
    // outputs the data in the wrong order. The first half of the spectrogram
    // comes out on the second half, and vice versa.
    for (int i = middle; i < NFFT; i++){
      float magnitude = sqrt(out[i].r * out[i].r + out[i].i * out[i].i);

      spectrogram[(w * NFFT) + (i - middle)] = magnitude;
    }

    for (int i = 0; i < middle; i++){
      float magnitude = sqrt(out[i].r * out[i].r + out[i].i * out[i].i);

      spectrogram[(w * NFFT) + (i + middle)] = magnitude;
    }
  }

  int scaleFactor = NUM_WINDOWS / TARGET_RESOLUTION;

  for (int i = 0; i < TARGET_RESOLUTION; i++){
    int start = i * scaleFactor;

    for (int j = 0; j < NFFT; j++){
      float sum = 0;

      for (int k = 0; k < scaleFactor; k++){
        sum += spectrogram[((start + k) * NFFT) + j];
      }

      downsampled[(i * TARGET_RESOLUTION) + j] = sum / scaleFactor;
    }
  }

  unsigned long timeEnd = millis(); 
  unsigned long duration = timeEnd - timeBegin;

  for (int w = 0; w < TARGET_RESOLUTION; w++){
    for (int i = 0; i < NFFT; i++){
      Serial.print(downsampled[(w * NFFT) + i]);

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
