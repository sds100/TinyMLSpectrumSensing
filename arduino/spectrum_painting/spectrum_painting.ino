// #include "TensorFlowLite.h"
// #include "tensorflow/lite/micro/all_ops_resolver.h"
// #include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/micro/micro_log.h"
// #include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"
#include "data.h"
#include "kiss_fft.h"

const uint16_t SAMPLES = 256;
const uint16_t NFFT = 256;
const float SAMPLING_FREQUENCY = 88000000;
const int NUM_WINDOWS = 1024;
const int TARGET_RESOLUTION = 64;

const int K = 3;
const int L = 16;
const int D = 4;

kiss_fft_cfg cfg;
kiss_fft_cpx in[SAMPLES];
kiss_fft_cpx out[NFFT];

float downsampled[NFFT * TARGET_RESOLUTION];

// const tflite::Model* model = nullptr;
// tflite::MicroInterpreter* interpreter = nullptr;
// TfLiteTensor* inputAugmented = nullptr;
// TfLiteTensor* inputPainted = nullptr;
// TfLiteTensor* output = nullptr;

// constexpr int tensor_arena_size = 50 * 1024;
// byte tensor_arena[tensor_arena_size] __attribute__((aligned(16)));

const int no_classes = 7;
const char* labels[no_classes] = {
  "z", "b", "w", "bw", "zb", "zw", "zbw"
};

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(4000);
  // wait for serial initialization so printing in setup works.
  while (!Serial)
    ;

  // model = tflite::GetModel(output_spectrum_painting_model_tflite);

  // if (model->version() != TFLITE_SCHEMA_VERSION) {
  //   MicroPrintf(
  //     "Model provided is schema version %d not equal "
  //     "to supported version %d.\n",
  //     model->version(), TFLITE_SCHEMA_VERSION);
  //   return;
  // }

  // tflite::AllOpsResolver resolver;

  // interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, tensor_arena_size);

  // TfLiteStatus allocate_status = interpreter->AllocateTensors();

  // if (allocate_status != kTfLiteOk) {
  //   Serial.println("ALLOCATE TENSORS FAILED");
  //   MicroPrintf("AllocateTensors() failed");
  // }

  // inputAugmented = interpreter->input(0);
  // inputPainted = interpreter->input(1);
  // output = interpreter->output(0);
}

void loop() {
  unsigned long timeBegin = millis();
  createDownsampledSpectrogram(real, imag);
  unsigned long timeDownsample = millis();

  // TODO: TAKE MIDDLE FREQUENCIES ONLY

  float* augmented = augment(downsampled);
  unsigned long timeAugment = millis();

  float* painted = paint(downsampled, augmented);
  unsigned long timePaint = millis();

  unsigned long timeTotal = millis();

  printSpectrogram(painted, TARGET_RESOLUTION, calculateNumAugmentedFreqBins(TARGET_RESOLUTION));
  Serial.println(timeDownsample - timeBegin);
  Serial.println(timeAugment - timeDownsample);
  Serial.println(timePaint - timeAugment);
  Serial.println(timeTotal  - timeBegin);

  while (true)
    ;
}

// int runInference(int8_t* augmented, int8_t* painted) {
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

  memcpy(downsampledCopy, downsampled, sizeof(downsampled));

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

void createDownsampledSpectrogram(const int8_t* real, const int8_t* imag) {
  kiss_fft_cfg cfg = kiss_fft_alloc(NFFT, false, NULL, NULL);

  float cumulative_row[NFFT];
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
      in[i].r = ((int8_t)pgm_read_byte(real + memIndex));
      in[i].i = ((int8_t)pgm_read_byte(imag + memIndex));
    }

    kiss_fft(cfg, in, out);

    int middle = NFFT / 2;

    // I'm not sure why but for my training data, computing the FFT puts
    // outputs the data in the wrong order. The first half of the spectrogram
    // comes out on the second half, and vice versa.
    for (int i = middle; i < NFFT; i++) {
      float magnitude = sqrt(out[i].r * out[i].r + out[i].i * out[i].i);

      cumulative_row[(i - middle)] += magnitude;
    }

    for (int i = 0; i < middle; i++) {
      float magnitude = sqrt(out[i].r * out[i].r + out[i].i * out[i].i);

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
}

void printSpectrogram(float* spectrogram, int timeBins, int freqBins) {
  for (int t = 0; t < timeBins; t++) {
    for (int f = 0; f < freqBins; f++) {
      Serial.print(spectrogram[(t * freqBins) + f]);

      if (f < freqBins - 1) {
        Serial.print(",");
      }
    }

    Serial.println();
  }
}