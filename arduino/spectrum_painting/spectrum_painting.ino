#include "TensorFlowLite.h"

#include "model.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_generated.h"

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* inputAugmented = nullptr;
TfLiteTensor* inputPainted = nullptr;
TfLiteTensor* output = nullptr;

constexpr int tensor_arena_size = 150 * 1024;
byte tensor_arena[tensor_arena_size] __attribute__((aligned(16)));

const int no_classes = 7;
const char* labels[no_classes] = {
  "z", "b", "w", "bw", "zb", "zw", "zbw"
};

void setup() {
  Serial.begin(9600);
  Serial.setTimeout(1000);
  // wait for serial initialization so printing in setup works.
  while (!Serial)
    ;

  pinMode(LED_BUILTIN, OUTPUT);

  model = tflite::GetModel(spectrum_painting_model_tflite);

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

  int inputLength = inputAugmented->bytes;

  while(!Serial.available()) ;
  int augmentedBytesRead = Serial.readBytes(inputAugmented->data.uint8, inputLength);
  int paintedBytesRead = Serial.readBytes(inputPainted->data.uint8, inputLength);

  // if (augmentedBytesRead != inputLength || paintedBytesRead != inputLength) {
  //   return;
  // }

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
