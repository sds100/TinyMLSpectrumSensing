#include "TensorFlowLite.h"

#include "model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "data.h"
#include "kiss_fft.h"

const uint16_t SAMPLES = 256;
const uint16_t NFFT = 256;
const float SAMPLING_FREQUENCY = 88000000;
const int NUM_WINDOWS = 128;
const int TARGET_RESOLUTION = 64;

const int K = 3;
const int L = 16;
const int D = 4;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* inputAugmented = nullptr;
TfLiteTensor* inputPainted = nullptr;
TfLiteTensor* output = nullptr;

constexpr int tensor_arena_size = 50 * 1024;
byte tensor_arena[tensor_arena_size] __attribute__((aligned(16)));

const int no_classes = 7;
const char* labels[no_classes] = {
  "z", "b", "w", "bw", "zb", "zw", "zbw"
};

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(4000);
  // wait for serial initialization so printing in setup works.
  while (!Serial)
    ;

  model = tflite::GetModel(output_spectrum_painting_model_tflite);

  // if (model->version() != TFLITE_SCHEMA_VERSION) {
  //   MicroPrintf(
  //     "Model provided is schema version %d not equal "
  //     "to supported version %d.\n",
  //     model->version(), TFLITE_SCHEMA_VERSION);
  //   return;
  // }

  tflite::AllOpsResolver resolver;

  interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, tensor_arena_size);

  TfLiteStatus allocate_status = interpreter->AllocateTensors();

  Serial.println(F("Allocated"));

  if (allocate_status != kTfLiteOk) {
    Serial.println("ALLOCATE TENSORS FAILED");
    MicroPrintf("AllocateTensors() failed");
  }

  inputAugmented = interpreter->input(0);
  inputPainted = interpreter->input(1);
  output = interpreter->output(0);
}

void loop() {
  unsigned long timeBegin = millis();
  float* downsampled = createDownsampledSpectrogram(real, imag);
  unsigned long timeDownsample = millis();

  float* augmented = augment(downsampled);
  uint8_t* digitizedAugmented = digitize(augmented);
  unsigned long timeAugment = millis();
  Serial.println(F("Augmented"));

  float* painted = paint(downsampled, augmented);
  uint8_t* digitizedPainted = digitize(painted);
  unsigned long timePaint = millis();

  // free(downsampled);
  // downsampled = nullptr;
  // free(augmented);
  // augmented = nullptr;
  // free(painted);
  // painted = nullptr;

  // size_t inputLength = inputAugmented->bytes;
  // Serial.println(inputLength);

  // int label = runInference(digitizedAugmented, digitizedPainted);
  size_t inputLength = inputAugmented->bytes;
  Serial.println(inputLength);
  Serial.println(inputAugmented->type);

  for (unsigned int i = 0; i < inputLength; i++) {
    inputAugmented->data.uint8[i] = (byte) digitizedAugmented[i];
  }

  for (unsigned int i = 0; i < inputLength; i++) {
    inputPainted->data.uint8[i] = (byte) digitizedPainted[i];
  }

  TfLiteStatus invoke_status = interpreter->Invoke();

  // if (invoke_status != kTfLiteOk) {
  //   Serial.println("Invoke failed " + String(invoke_status));
  //   // return -1;
  // }

  // int index_loc_highest_prob = -1;
  // float highest_prob = -1.0;

  // for (int i = 0; i < no_classes; i++) {
  //   if (output->data.uint8[i] > highest_prob) {
  //     highest_prob = output->data.uint8[i];
  //     index_loc_highest_prob = i;
  //   }
  // }

  unsigned long timeInference = millis();

  unsigned long timeTotal = millis();

  // printSpectrogram(digitizedPainted, TARGET_RESOLUTION, calculateNumAugmentedFreqBins(TARGET_RESOLUTION));

  int timeBins = TARGET_RESOLUTION;
  int freqBins = calculateNumAugmentedFreqBins(TARGET_RESOLUTION);

  for (int t = 0; t < timeBins; t++) {
    for (int f = 0; f < freqBins; f++) {
      Serial.print(digitizedPainted[(t * freqBins) + f]);

      if (f < freqBins - 1) {
        Serial.print(F(","));
      }
    }

    Serial.println();
  }

  Serial.println(timeDownsample - timeBegin);
  Serial.println(timeAugment - timeDownsample);
  Serial.println(timePaint - timeAugment);
  Serial.println(timeInference - timePaint);
  Serial.println(timeTotal - timeBegin);
  // Serial.println(index_loc_highest_prob);

  while (true)
    ;
}

// int runInference(uint8_t* augmented, uint8_t* painted) {
//   size_t inputLength = inputAugmented->bytes;

//   for (unsigned int i = 0; i < inputLength; i++) {
//     inputAugmented->data.uint8[i] = augmented[i];
//   }

//   for (unsigned int i = 0; i < inputLength; i++) {
//     inputPainted->data.uint8[i] = painted[i];
//   }

//   TfLiteStatus invoke_status = interpreter->Invoke();

//   if (invoke_status != kTfLiteOk) {
//     Serial.println("Invoke failed " + String(invoke_status));
//     return -1;
//   }

//   int index_loc_highest_prob = -1;
//   float highest_prob = -1.0;

//   for (int i = 0; i < no_classes; i++) {
//     if (output->data.uint8[i] > highest_prob) {
//       highest_prob = output->data.uint8[i];
//       index_loc_highest_prob = i;
//     }
//   }

//   return index_loc_highest_prob;
// }

float* createDownsampledSpectrogram(const int8_t* real, const int8_t* imag) {
  // DOES LOADING DATA INTO MEMORY SPEED IT UP?
  kiss_fft_cpx fftIn[SAMPLES];
  kiss_fft_cpx fftOut[NFFT];

  kiss_fft_cfg cfg = kiss_fft_alloc(NFFT, false, NULL, NULL);

  float cumulative_row[NFFT];
  float* downsampled = (float*)calloc(NFFT * TARGET_RESOLUTION, sizeof(float));

  int scaleFactor = NUM_WINDOWS / TARGET_RESOLUTION;

  int downsampledRowCounter = 0;

  int middleFreq = NFFT / 2;
  int startFreq = middleFreq - 32;

  for (int w = 0; w < NUM_WINDOWS; w++) {
    if (w % scaleFactor == 0) {
      for (int i = 0; i < NFFT; i++) {
        cumulative_row[i] = 0;
      }
    }

    for (int i = 0; i < SAMPLES; i++) {
      int memIndex = (w * SAMPLES) + i;

      // Don't need to rescale the data. Doing FFT on integers works fine.
      fftIn[i].r = ((int8_t)pgm_read_byte(real + memIndex));
      fftIn[i].i = ((int8_t)pgm_read_byte(imag + memIndex));
    }

    kiss_fft(cfg, fftIn, fftOut);

    int middle = NFFT / 2;

    // I'm not sure why but for my training data, computing the FFT puts
    // outputs the data in the wrong order. The first half of the spectrogram
    // comes out on the second half, and vice versa.
    for (int i = middle; i < NFFT; i++) {
      float magnitude = sqrt(fftOut[i].r * fftOut[i].r + fftOut[i].i * fftOut[i].i);

      cumulative_row[(i - middle)] += magnitude;
    }

    for (int i = 0; i < middle; i++) {
      float magnitude = sqrt(fftOut[i].r * fftOut[i].r + fftOut[i].i * fftOut[i].i);

      cumulative_row[(i + middle)] += magnitude;
    }

    if (w != 0 && (w + 1) % scaleFactor == 0) {
      for (int i = 0; i < NFFT; i++) {
        cumulative_row[i] = cumulative_row[i] / scaleFactor;
      }

      // Only take the frequencies that are filled by the Wi-Fi signal.

      memcpy(downsampled + (downsampledRowCounter * TARGET_RESOLUTION), cumulative_row + startFreq, TARGET_RESOLUTION * sizeof(float));
      downsampledRowCounter += 1;
    }
  }

  kiss_fft_free(cfg);
  return downsampled;
}

int calculateNumAugmentedFreqBins(int freqBins) {
  return ((freqBins - L) / D) + 1;
}

float* augment(float* in) {
  // The number of "columns", i.e frequency bins in each time window.
  int freqBins = TARGET_RESOLUTION;

  // The number of rows in the spectrogram - i.e number of time bins.
  int timeBins = TARGET_RESOLUTION;

  int augmentedFreqBins = calculateNumAugmentedFreqBins(freqBins);
  int outLength = augmentedFreqBins * timeBins;

  float downsampledCopy[TARGET_RESOLUTION * TARGET_RESOLUTION];

  memcpy(downsampledCopy, in, sizeof(in));

  float* out = (float*)calloc(outLength, sizeof(float));

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

  return out;
}

float* paint(float* downsampled, float* augmented) {
  // The number of "columns", i.e frequency bins in each time window.
  int freqBins = TARGET_RESOLUTION;

  // The number of rows in the spectrogram - i.e number of time bins.
  int timeBins = TARGET_RESOLUTION;

  int augmentedFreqBins = calculateNumAugmentedFreqBins(freqBins);
  int outLength = augmentedFreqBins * timeBins;
  float* out = (float*)calloc(outLength, sizeof(float));

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

  return out;
}

uint8_t* digitize(float* in) {
  // The number of rows in the spectrogram - i.e number of time bins.
  int timeBins = TARGET_RESOLUTION;
  int freqBins = calculateNumAugmentedFreqBins(TARGET_RESOLUTION);

  int outLength = freqBins * timeBins;
  uint8_t* out = (uint8_t*)calloc(outLength, sizeof(uint8_t));
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
    return out;
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

  return out;
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

void printSpectrogram(uint8_t* spectrogram, int timeBins, int freqBins) {
  for (int t = 0; t < timeBins; t++) {
    for (int f = 0; f < freqBins; f++) {
      Serial.print(F(spectrogram[(t * freqBins) + f]));

      if (f < freqBins - 1) {
        Serial.print(F(","));
      }
    }

    Serial.println();
  }
}