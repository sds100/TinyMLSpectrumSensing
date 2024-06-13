#include "TensorFlowLite.h"

#include "model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "data.h"
#include "kiss_fft.h"

const int NUM_WINDOWS = 256;
const uint16_t SAMPLES = 256;
const uint16_t NFFT = 256;
const int TARGET_RESOLUTION = 64;

const int K = 3;
const int L = 16;
const int D = 4;

kiss_fft_cpx fftIn[SAMPLES];
kiss_fft_cpx fftOut[NFFT];

kiss_fft_cfg kssCfg;

float downsampled[TARGET_RESOLUTION * TARGET_RESOLUTION];
float downsampledCopy[TARGET_RESOLUTION * TARGET_RESOLUTION];

float augmented[13 * TARGET_RESOLUTION];
uint8_t digitizedAugmented[13 * TARGET_RESOLUTION];

float painted[13 * TARGET_RESOLUTION];
uint8_t digitizedPainted[13 * TARGET_RESOLUTION];

int8_t realBuffer[SAMPLES];
int8_t imagBuffer[SAMPLES];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* inputAugmented = nullptr;
TfLiteTensor* inputPainted = nullptr;
TfLiteTensor* output = nullptr;

constexpr int tensor_arena_size = 50 * 1024;
byte tensor_arena[tensor_arena_size] __attribute__((aligned(16)));

const int no_classes = 7;

void setup() {
  Serial.begin(115200);
  // wait for someone to connect to serial before printing.
  while (!Serial)
    ;

  model = tflite::GetModel(output_spectrum_painting_model_tflite);

  tflite::AllOpsResolver resolver;

  interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, tensor_arena_size);

  interpreter->AllocateTensors();

  inputAugmented = interpreter->input(0);
  inputPainted = interpreter->input(1);
  output = interpreter->output(0);

  kssCfg = kiss_fft_alloc(NFFT, false, NULL, NULL);
}

void loop() {
  unsigned long timeBegin = millis();

  createDownsampledSpectrogram(real, imag, downsampled);
  unsigned long timeDownsample = millis();

  augment(downsampled, augmented);
  unsigned long timeAugment = millis();

  paint(downsampled, augmented, painted);
  unsigned long timePaint = millis();

  digitize(augmented, digitizedAugmented);
  digitize(painted, digitizedPainted);
  unsigned long timeDigitize = millis();

  size_t inputLength = inputAugmented->bytes;

  for (unsigned int i = 0; i < inputLength; i++) {
    inputAugmented->data.uint8[i] = digitizedAugmented[i];
  }

  for (unsigned int i = 0; i < inputLength; i++) {
    inputPainted->data.uint8[i] = digitizedPainted[i];
  }

  TfLiteStatus invoke_status = interpreter->Invoke();

  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed " + String(invoke_status));
    return;
  }

  int predictedLabel = -1;
  float highestProbability = -1.0;

  for (int i = 0; i < no_classes; i++) {
    if (output->data.uint8[i] > highestProbability) {
      highestProbability = output->data.uint8[i];
      predictedLabel = i;
    }
  }

  unsigned long timeInference = millis();

  unsigned long timeTotal = millis();

  int timeBins = TARGET_RESOLUTION;
  int freqBins = calculateNumAugmentedFreqBins(TARGET_RESOLUTION);

  for (int t = 0; t < timeBins; t++) {
    for (int f = 0; f < freqBins; f++) {
      Serial.print((uint8_t)digitizedPainted[(t * freqBins) + f]);

      if (f < freqBins - 1) {
        Serial.print(F(","));
      }
    }

    Serial.println();
  }

  Serial.println(timeDownsample - timeBegin);
  Serial.println(timeAugment - timeDownsample);
  Serial.println(timePaint - timeAugment);
  Serial.println(timeDigitize - timePaint);
  Serial.println(timeInference - timeDigitize);

  Serial.println(timeTotal - timeBegin);
  Serial.println(predictedLabel);

  while (true)
    ;
}

void createDownsampledSpectrogram(const int8_t* real, const int8_t* imag, float* out) {
  // DOES LOADING DATA INTO MEMORY SPEED IT UP?
  float cumulativeRows[TARGET_RESOLUTION];

  int timeScaleFactor = NUM_WINDOWS / TARGET_RESOLUTION;
  int freqScaleFactor = NFFT / TARGET_RESOLUTION;

  int downsampledRowCounter = 0;

  for (int w = 0; w < NUM_WINDOWS; w++) {
    // If we reached the next step in downsampling the rows
    // then set the sum to 0.
    if (w % timeScaleFactor == 0) {
      for (int i = 0; i < TARGET_RESOLUTION; i++) {
        cumulativeRows[i] = 0;
      }
    }

    int memIndex = w * SAMPLES;

    memcpy_P(realBuffer, real + memIndex, SAMPLES);
    memcpy_P(imagBuffer, imag + memIndex, SAMPLES);

    for (int i = 0; i < SAMPLES; i++) {
      fftIn[i] = {realBuffer[i], imagBuffer[i]};
    }

    kiss_fft(kssCfg, fftIn, fftOut);

    int middle = TARGET_RESOLUTION / 2;

    // I'm not sure why but for my training data, computing the FFT puts
    // outputs the data in the wrong order. The first half of the spectrogram
    // comes out on the second half, and vice versa so compute each
    // half of the spectrogram separately and switch the order.
    for (int i = middle; i < TARGET_RESOLUTION; i++) {
      // Downsample the frequency bins to the target resolution.
      float meanFreq = 0;

      for (int j = 0; j < freqScaleFactor; j++){
        int index = (i * freqScaleFactor) + j;
        float magnitude = sqrt(sq(fftOut[index].r) + sq(fftOut[index].i));
        meanFreq += magnitude;
      }

      meanFreq /= freqScaleFactor;

      cumulativeRows[(i - middle)] += meanFreq;
    }

    for (int i = 0; i < middle; i++) {
      // Downsample the frequency bins to the target resolution.
      float meanFreq = 0;

      for (int j = 0; j < freqScaleFactor; j++){
        int index = (i * freqScaleFactor) + j;
        float magnitude = sqrt(sq(fftOut[index].r) + sq(fftOut[index].i));
        meanFreq += magnitude;
      }

      meanFreq /= freqScaleFactor;

      cumulativeRows[(i + middle)] += meanFreq;
    }

    // if reached the end of processing a group of rows to
    // downsample.
    if (w != 0 && (w + 1) % timeScaleFactor == 0) {
      // Downsample each freq bin the time row.
      for (int i = 0; i < TARGET_RESOLUTION; i++) {
        cumulativeRows[i] = cumulativeRows[i] / timeScaleFactor;
      }

      memcpy(out + (downsampledRowCounter * TARGET_RESOLUTION), cumulativeRows, TARGET_RESOLUTION * sizeof(float));
      downsampledRowCounter += 1;
    }
  }
}

int calculateNumAugmentedFreqBins(int freqBins) {
  return ((freqBins - L) / D) + 1;
}

void augment(float* in, float* out) {
  // The number of "columns", i.e frequency bins in each time window.
  int freqBins = TARGET_RESOLUTION;

  // The number of rows in the spectrogram - i.e number of time bins.
  int timeBins = TARGET_RESOLUTION;

  int augmentedFreqBins = calculateNumAugmentedFreqBins(freqBins);

  float input_mean = 0;  // The mean value in the whole spectrogram.

  for (int i = 0; i < freqBins * timeBins; i++) {
    input_mean += in[i];
  }

  input_mean /= (freqBins * timeBins);

  for (int t = 0; t < timeBins; t++) {
    int f_augmented = 0;
    int f = 0;

    while (f <= freqBins - L) {
      float window[L];
      int startOfWindow = t * freqBins + f;

      for (int i = 0; i < L; i++) {
        window[i] = in[startOfWindow + i];
      }

      insertionSort(window, L);

      float meanTopK = 0;

      // IS IT SORTED IN ASCENDING OR DESCENDING ORDER???
      for (int i = 0; i < K; i++) {
        meanTopK += window[(L - 1) - i];
      }

      meanTopK /= K;

      in[(t * freqBins) + f] = meanTopK;

      float value = in[(t * freqBins) + f] - input_mean;
      // clip the value at 0 because subtracting the mean can
      // sometimes give negative values
      out[(t * augmentedFreqBins) + f_augmented] = max(0, value);

      f_augmented++;
      f += D;
    }
  }
}

void paint(float* downsampled, float* augmented, float* out) {
  // The number of "columns", i.e frequency bins in each time window.
  int freqBins = TARGET_RESOLUTION;

  // The number of rows in the spectrogram - i.e number of time bins.
  int timeBins = TARGET_RESOLUTION;

  int augmentedFreqBins = calculateNumAugmentedFreqBins(freqBins);

  for (int t = 0; t < timeBins; t++) {
    // calculate the average value of the time bin
    // in the downsampled spectrogram
    float meanTimeOriginal = 0;

    for (int f = 0; f < freqBins; f++) {
      meanTimeOriginal += downsampled[(t * freqBins) + f];
    }

    meanTimeOriginal /= freqBins;

    for (int f = 0; f < augmentedFreqBins; f++) {
      float paintedValue = augmented[(t * augmentedFreqBins) + f] - meanTimeOriginal;

      out[(t * augmentedFreqBins) + f] = max(0, paintedValue);
    }
  }
}

void digitize(float* in, uint8_t* out) {
  // The number of rows in the spectrogram - i.e number of time bins.
  int timeBins = TARGET_RESOLUTION;
  int freqBins = calculateNumAugmentedFreqBins(TARGET_RESOLUTION);

  float maxValue = 0;

  for (int t = 0; t < timeBins; t++) {
    for (int f = 0; f < freqBins; f++) {
      float value = in[(t * freqBins) + f];
      maxValue = max(maxValue, value);
    }
  }

  // if the max value is 0 then there is no data to scale so
  // just return a spectrogram with all zeros.
  if (maxValue == 0) {
    for (int i = 0; i < timeBins * freqBins; i++) {
      out[i] = 0;
    }
    return;
  }

  // We want to store the values in one byte so 255 is the max value.
  float scaleFactor = 255 / maxValue;

  for (int t = 0; t < timeBins; t++) {
    for (int f = 0; f < freqBins; f++) {
      int index = (t * freqBins) + f;
      uint8_t value = (uint8_t)(in[index] * scaleFactor);

      out[index] = value;
    }
  }
}

/**
  From https://github.com/bxparks/AceSorting
**/
void insertionSort(float data[], uint16_t n) {
  for (uint16_t i = 1; i < L; i++) {
    float temp = data[i];

    // Shift one slot to the right.
    uint16_t j;
    for (j = i; j > 0; j--) {
      if (data[j - 1] <= temp) break;
      data[j] = data[j - 1];
    }

    // This can assign 'temp' back into the original slot if no shifting was
    // done. That's ok because T is assumed to be relatively cheap to copy, and
    // checking for (i != j) is more expensive than just doing the extra
    // assignment.
    data[j] = temp;
  }
}