#include "model.h"
#include "data.h"

#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "kiss_fft.h"

const uint16_t SAMPLES = 256;
const uint16_t NFFT = 64;
const float SAMPLING_FREQUENCY = 88000000;
const int NUM_WINDOWS = 1024;
const int TARGET_RESOLUTION = 64;

kiss_fft_cfg cfg;
kiss_fft_cpx in[SAMPLES];
kiss_fft_cpx out[NFFT];

float downsampled[NFFT * TARGET_RESOLUTION];

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
  Serial.begin(115200);
  Serial.setTimeout(4000);
  // wait for serial initialization so printing in setup works.
  while (!Serial)
    ;

  model = tflite::GetModel(output_spectrum_painting_model_tflite);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
      "Model provided is schema version %d not equal "
      "to supported version %d.\n",
      model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  tflite::AllOpsResolver resolver;

  interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, tensor_arena_size);

  TfLiteStatus allocate_status = interpreter->AllocateTensors();

  if (allocate_status != kTfLiteOk) {
    Serial.println("ALLOCATE TENSORS FAILED");
    MicroPrintf("AllocateTensors() failed");
  }

  inputAugmented = interpreter->input(0);
  inputPainted = interpreter->input(1);
  output = interpreter->output(0);
}

void loop() {
  createDownsampledSpectrogram(real, imag);
  printSpectrogram(downsampled);

  printSpectrogram(downsampled);
}

int runInference(int8_t* augmented, int8_t* painted) {
  size_t inputLength = inputAugmented->bytes;

  for (unsigned int i = 0; i < inputLength; i++) {
    inputAugmented->data.uint8[i] = augmented[i];
  }

  for (unsigned int i = 0; i < inputLength; i++) {
    inputPainted->data.uint8[i] = painted[i];
  }

  TfLiteStatus invoke_status = interpreter->Invoke();

  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed " + String(invoke_status));
    return -1;
  }

  int index_loc_highest_prob = -1;
  float highest_prob = -1.0;

  for (int i = 0; i < no_classes; i++) {
    if (output->data.uint8[i] > highest_prob) {
      highest_prob = output->data.uint8[i];
      index_loc_highest_prob = i;
    }
  }

  return index_loc_highest_prob;
}

void createDownsampledSpectrogram(const int8_t* real, const int8_t* imag) {
  kiss_fft_cfg cfg = kiss_fft_alloc(NFFT, false, NULL, NULL);

  float cumulative_row[NFFT];
  int scaleFactor = NUM_WINDOWS / TARGET_RESOLUTION;

  int downsampledRowCounter = 0;

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

      memcpy(downsampled + (downsampledRowCounter * TARGET_RESOLUTION), cumulative_row, sizeof(cumulative_row));
      downsampledRowCounter += 1;
    }
  }
}

void printSpectrogram(float* spectrogram) {
  for (int w = 0; w < TARGET_RESOLUTION; w++) {
    for (int i = 0; i < NFFT; i++) {
      Serial.print(spectrogram[(w * NFFT) + i]);

      if (i < NFFT - 1) {
        Serial.print(",");
      }
    }

    Serial.println();
  }
}