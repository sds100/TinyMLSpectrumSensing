import json
import os
import sys
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import tensorflow as tf

import spectrum_painting_data as sp_data
import spectrum_painting_model as sp_model
import spectrum_painting_predict as sp_predict
import spectrum_painting_training as sp_training
from spectrum_painting_result import SpectrumPaintingResult, ModelResult

gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def calc_accuracy(y_test, predictions) -> float:
    return np.mean(np.asarray(y_test) == np.asarray(predictions))


run_name_arg: Optional[str] = None
filters: int = 2
spectrogram_count: int = -1

if len(sys.argv) > 1:
    run_name_arg = sys.argv[1]

if len(sys.argv) > 2:
    filters = int(sys.argv[2])

if len(sys.argv) > 3:
    spectrogram_count = int(sys.argv[3])

print(f"This run is called '{run_name_arg}'")
print(f"Using {filters} filters")
print(f"Creating {spectrogram_count} spectrograms for each SNR and class")

classes = ["Z", "B", "W", "BW", "ZB", "ZW", "ZBW"]
snr_list = [0, 5, 10, 15, 20, 25, 30]

spectrum_painting_options = sp_training.SpectrumPaintingTrainingOptions(
    downsample_resolution=64,
    k=3,
    l=16,
    d=4
)

training_count = 10

results: Dict[int, SpectrumPaintingResult] = {}

for snr in snr_list:
    results[snr] = SpectrumPaintingResult(
        snr=snr,
        full_model_results=[],
        lite_model_results=[],
        label_names=[]
    )

print("Loading spectrograms")
# Create the spectrograms once.
spectrograms = sp_data.load_spectrograms(data_dir="data/numpy",
                                         classes=classes,
                                         snr_list=snr_list,
                                         windows_per_spectrogram=256,
                                         window_length=256,
                                         nfft=64,
                                         spectrogram_count=spectrogram_count)

# Create 10 models, and run inference for each SNR once on each model.
for i in range(training_count):
    run_name: str

    if run_name_arg is None:
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        run_name = run_name_arg

    print(f"Starting iteration {i}")
    print("Splitting training and test data")

    train_test_sets = sp_training.create_spectrum_painting_train_test_sets(
        spectrograms=spectrograms,
        label_names=classes,
        options=spectrum_painting_options,
        test_size=0.3
    )

    print(f"Number of training images: {len(train_test_sets.y_train)}")
    print(f"Number of testing images: {len(train_test_sets.y_test)}")

    image_shape = train_test_sets.x_train_augmented[0].shape

    full_model = sp_model.create_tensorflow_model(image_shape=image_shape,
                                                  label_count=len(train_test_sets.label_names),
                                                  filters=filters)

    sp_model.fit_model(full_model, train_test_sets, epochs=100, early_stopping_patience=10)

    output_file = f"output/spectrum-painting-model-{run_name}.keras"
    full_model.save(output_file, save_format="keras")
    full_model_size = os.stat(output_file).st_size

    print("\n")

    lite_model = sp_model.convert_to_tensorflow_lite(full_model,
                                                     train_test_sets.x_test_augmented,
                                                     train_test_sets.x_test_painted)
    lite_model_size = len(lite_model)

    lite_output_file = f"output/spectrum-painting-model-{run_name}.tflite"

    with open(lite_output_file, "wb") as f:
        f.write(lite_model)

    no_quantization_model = sp_model.convert_to_tensorflow_lite_no_quantization(full_model)
    no_quantization_file = f"output/spectrum-painting-model-{run_name}-no-quant.tflite"

    with open(no_quantization_file, "wb") as f:
        f.write(no_quantization_model)

    for snr in snr_list:
        print(f"Testing SNR: {snr}")

        test_indices = np.argwhere(train_test_sets.test_snr == snr).squeeze()
        test_labels = train_test_sets.y_test[test_indices]
        test_augmented = train_test_sets.x_test_augmented[test_indices]
        test_painted = train_test_sets.x_test_painted[test_indices]

        full_model_predictions = [sp_predict.predict_full_model(full_model, x_a, x_p) for (x_a, x_p) in
                                  zip(test_augmented, test_painted)]
        lite_model_predictions = [sp_predict.predict_lite_model(lite_model, x_a, x_p) for (x_a, x_p) in
                                  zip(test_augmented, test_painted)]

        full_model_result = ModelResult(
            labels=test_labels.astype(int).tolist(),
            predictions=full_model_predictions,
            size=full_model_size
        )

        lite_model_result = ModelResult(
            labels=test_labels.astype(int).tolist(),
            predictions=lite_model_predictions,
            size=lite_model_size
        )

        results[snr].label_names = train_test_sets.label_names
        results[snr].full_model_results.append(full_model_result)
        results[snr].lite_model_results.append(lite_model_result)

        print(f"Full model accuracy = {calc_accuracy(test_labels, full_model_predictions)}")
        print(f"Lite model accuracy = {calc_accuracy(test_labels, lite_model_predictions)}")

    print("Saving results")
    with open(f"output/results-{run_name}.json", "w") as f:
        result_list = []

        for (snr, result) in results.items():
            result_list.append(result.to_dict())

        json.dump({"results": result_list}, f)
