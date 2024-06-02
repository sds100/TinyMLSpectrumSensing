#include "TensorFlowLite.h"

#include "model.h"
#include "augmented_image.h"
#include "painted_image.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

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
  while (!Serial);

  pinMode(LED_BUILTIN, OUTPUT);

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
//   inputPainted = interpreter->input(1);
  output = interpreter->output(0);
}

void loop() {

  size_t inputLength = inputAugmented->bytes;

  for (int i = 0; i < output_augmented_image_bytes_len; i++) {
    inputAugmented->data.uint8[i] = output_augmented_image_bytes[i];
  }

//   for (int i = 0; i < output_painted_image_bytes_len; i++) {
//     inputPainted->data.uint8[i] = output_painted_image_bytes[i];
//   }

  unsigned long timeBegin = millis();
  TfLiteStatus invoke_status = interpreter->Invoke();
  unsigned long timeEnd = millis();

  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed " + String(invoke_status));
    return;
  }

  int index_loc_highest_prob = -1;
  float highest_prob = -1.0;

  for (int i = 0; i < no_classes; i++) {
    if (output->data.uint8[i] > highest_prob) {
      highest_prob = output->data.uint8[i];
      index_loc_highest_prob = i;
    }
  }

  unsigned long duration = timeEnd - timeBegin;

  Serial.println(index_loc_highest_prob);
  Serial.println(duration);
}
