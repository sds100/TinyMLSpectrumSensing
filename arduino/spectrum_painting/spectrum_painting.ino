#include "TensorFlowLite.h"

#include "model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

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

kiss_fft_cpx fftIn[SAMPLES];
kiss_fft_cpx fftOut[NFFT];

kiss_fft_cfg kssCfg;

float downsampled[TARGET_RESOLUTION * TARGET_RESOLUTION];
float downsampledCopy[TARGET_RESOLUTION * TARGET_RESOLUTION];

float augmented[13 * TARGET_RESOLUTION];
char digitizedAugmented[13 * TARGET_RESOLUTION];

float painted[13 * TARGET_RESOLUTION];
char digitizedPainted[13 * TARGET_RESOLUTION];

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

  TfLiteStatus allocate_status = interpreter->AllocateTensors();

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
  digitize(augmented, digitizedAugmented);
  unsigned long timeAugment = millis();

  paint(downsampled, augmented, painted);
  digitize(painted, digitizedPainted);
  unsigned long timePaint = millis();

  // int predictedLabel = runInference(digitizedAugmented, digitizedPainted);
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

  // printSpectrogram(digitizedPainted, TARGET_RESOLUTION, calculateNumAugmentedFreqBins(TARGET_RESOLUTION));

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
  Serial.println(timeInference - timePaint);
  Serial.println(timeTotal - timeBegin);
  Serial.println(predictedLabel);

  while (true)
    ;
}

void createDownsampledSpectrogram(const int8_t* real, const int8_t* imag, float* out) {
  // DOES LOADING DATA INTO MEMORY SPEED IT UP?
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
      fftIn[i].r = ((int8_t)pgm_read_byte(real + memIndex));
      fftIn[i].i = ((int8_t)pgm_read_byte(imag + memIndex));
    }

    kiss_fft(kssCfg, fftIn, fftOut);

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

      memcpy(out + (downsampledRowCounter * TARGET_RESOLUTION), cumulative_row + startFreq, TARGET_RESOLUTION * sizeof(float));
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

void digitize(float* in, char* out) {
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
      char value = (char)(in[index] * scaleFactor);

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