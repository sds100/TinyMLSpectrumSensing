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

constexpr int tensor_arena_size = 150 * 1024;
byte tensor_arena[tensor_arena_size] __attribute__((aligned(16)));

const int no_classes = 7;
const char* labels[no_classes] = {
  "z", "b", "w", "bw", "zb", "zw", "zbw"
};

void setup() {
  Serial.begin(9600);
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
  unsigned long timeBegin = millis();

  Serial.print("Input 1 details: type: ");
  Serial.print(inputAugmented->type);
  Serial.print(", shape: (");
  for (int i = 0; i < inputAugmented->dims->size; ++i) {
    Serial.print(inputAugmented->dims->data[i]); 
    if (i < inputAugmented->dims->size - 1) {
      Serial.print(", ");
    }
  }
  Serial.println(")");
  Serial.println(String(inputAugmented->bytes));

  // Serial.print("Input Augmented details: type: ");
  // Serial.println(inputAugmented->type);
  // Serial.print("Input Painted details: type: ");
  // Serial.println(inputPainted->type);

  // Serial.print("Input shape: (");
  // for (int i = 0; i < inputAugmented->dims->size; ++i) {
  //   Serial.print(inputAugmented->dims->data[i]);
  //   if (i < inputAugmented->dims->size - 1) {
  //     Serial.print(", ");
  //   }
  // }
  // Serial.println(")");

  // put your main code here, to run repeatedly:
  // int bytesRead = Serial.readBytes(inputAugmented->data.raw, 27360);
  // int bytesReadPainted = Serial.readBytes(inputPainted->data.raw, 27360);

  for (int i = 0; i < augmented_image_bytes_len; i++) {
    inputAugmented->data.uint8[i] = augmented_image_bytes[i];
  }

  for (int i = 0; i < painted_image_bytes_len; i++) {
    inputPainted->data.uint8[i] = painted_image_bytes[i];
  }

  // Serial.println(String(bytesRead));
  // Serial.println(String(bytesReadPainted));

  TfLiteStatus invoke_status = interpreter->Invoke();

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

  Serial.print("Predicted class: ");
  Serial.println(index_loc_highest_prob);
  Serial.println(labels[index_loc_highest_prob]);
  Serial.println("with probability:");
  Serial.println(highest_prob);

  //execution time calculation
  unsigned long timeEnd = millis();
  unsigned long duration = timeEnd - timeBegin;
  Serial.print("Duration (ms): ");
  Serial.println(duration);
  Serial.println();
}
