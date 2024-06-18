import json
import os
from datetime import datetime
from typing import Dict

import numpy as np
import tensorflow as tf

import spectrum_painting_data as sp_data
import spectrum_painting_model as sp_model
import spectrum_painting_predict as sp_predict
import spectrum_painting_training as sp_training
from spectrum_painting_result import SpectrumPaintingResult, ModelResult

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def calc_accuracy(y_test, predictions) -> float:
    return np.mean(np.asarray(y_test) == np.asarray(predictions))


classes = ["Z", "B", "W", "BW", "ZB", "ZW", "ZBW"]
snr_list = [-100, -15, -10, -5, 0, 5, 10, 15, 20, 30]

spectrum_painting_options = sp_training.SpectrumPaintingTrainingOptions(
    downsample_resolution=64,
    k=3,
    l=16,
    d=4
)

training_count = 1

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
                                         nfft=64)

# Create 10 models, and run inference for each SNR once on each model.
for i in range(training_count):
    print(f"Starting iteration {i}")
    print("Splitting training and test data")

    train_test_sets = sp_training.create_spectrum_painting_train_test_sets(
        spectrograms=spectrograms,
        label_names=classes,
        options=spectrum_painting_options,
        test_size=0.3
    )

    image_shape = train_test_sets.x_train_augmented[0].shape

    full_model = sp_model.create_tensorflow_model(image_shape=image_shape,
                                                  label_count=len(train_test_sets.label_names))

    sp_model.fit_model(full_model, train_test_sets, epochs=200, early_stopping_patience=20)

    output_file = f"output/spectrum-painting-model.keras"
    full_model.save(output_file, save_format="keras")
    full_model_size = os.stat(output_file).st_size

    print("\n")

    lite_model = sp_model.convert_to_tensorflow_lite(full_model,
                                                     train_test_sets.x_test_augmented,
                                                     train_test_sets.x_test_painted)
    lite_model_size = len(lite_model)

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
            labels=test_labels,
            predictions=full_model_predictions,
            size=full_model_size
        )

        lite_model_result = ModelResult(
            labels=test_labels,
            predictions=lite_model_predictions,
            size=lite_model_size
        )

        results[snr].label_names = train_test_sets.label_names
        results[snr].full_model_results.append(full_model_result)
        results[snr].lite_model_results.append(lite_model_result)

        print(f"Full model accuracy = {calc_accuracy(test_labels, full_model_predictions)}")
        print(f"Lite model accuracy = {calc_accuracy(test_labels, lite_model_predictions)}")

    print("Saving model")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(f"output/results-{timestamp}.json", "w") as f:
        json.dump({"results": results}, f)
