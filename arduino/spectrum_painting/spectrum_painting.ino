// #include "TensorFlowLite.h"
// #include "tensorflow/lite/micro/all_ops_resolver.h"
// #include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/micro/micro_log.h"
// #include "tensorflow/lite/schema/schema_generated.h"

#include "model.h"
#include "data.h"
#include "kiss_fft.h"

const uint16_t SAMPLES = 256;
const uint16_t NFFT = 64;
const float SAMPLING_FREQUENCY = 88000000;
const int NUM_WINDOWS = 256;
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
  
  // float* augmented = augment(downsampled);
  unsigned long timeAugment = millis();

  printSpectrogram(downsampled);
  Serial.println(timeAugment - timeBegin);

  while(true);
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



float* augment(float* in){
  int freqBins = NFFT;
  int timeBins = TARGET_RESOLUTION;

  int augmented_width = ((freqBins - L) / D) + 1;
  int outLength = augmented_width * timeBins;

  float* out = (float*) calloc(outLength, sizeof(float));

  float input_mean = 0.0; // The mean value in the whole spectrogram.

  for (int i = 0; i < freqBins * timeBins; i++){
    input_mean += in[i];
  }

  input_mean /= (freqBins * timeBins);

  for (int t = 1; t < timeBins; t++){
    int f_augmented = 0;
    int f = 0;

    while (f <= freqBins - L){
      float window[L];
      float* startOfWindow = in + (t * timeBins + f);
      memcpy(window, startOfWindow, L);

      quickSortMiddle(window, L);

      float meanTopK = 0;

      // IS IT SORTED IN ASCENDING OR DESCENDING ORDER???
      for (int i = 0; i < K; i++){
        meanTopK += window[i];
      }

      meanTopK /= K;

      in[t * timeBins + f] = meanTopK;

      out[(t * timeBins) + f_augmented] = in[(t * timeBins) + f] - input_mean;

      f_augmented++;
      f += D;
    }
  }

  return out;
}

void paint(float* out){

}

/**
  From https://github.com/bxparks/AceSorting
**/
void quickSortMiddle(float data[], uint16_t n) {
  if (n <= 1) return;

  float pivot = data[n / 2];
  float* left = data;
  float* right = data + n - 1;

  while (left <= right) {
    if (*left < pivot) {
      left++;
    } else if (pivot < *right) {
      right--;
    } else {
      swap(*left, *right);
      left++;
      right--;
    }
  }

  quickSortMiddle(data, right - data + 1);
  quickSortMiddle(left, data + n - left);
}

void swap(float& a, float& b) {
    float temp = a;
    a = b;
    b = temp;
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