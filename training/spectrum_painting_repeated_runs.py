import json
from datetime import datetime
from typing import List, Dict

import numpy as np
import tensorflow as tf

import spectrum_painting_data as sp_data
import spectrum_painting_model as sp_model
import spectrum_painting_predict as sp_predict
import spectrum_painting_training as sp_training
from result import Result

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

training_count = 10

results: List[Dict] = []

for snr in snr_list:
    print(f"Starting snr {snr}")
    print("Loading spectrograms")
    spectrograms = sp_data.load_spectrograms(data_dir="data/numpy",
                                             classes=classes,
                                             snr_list=[snr],
                                             windows_per_spectrogram=256,
                                             window_length=256,
                                             nfft=64)

    full_model_predictions: List[List[int]] = []
    full_model_labels: List[List[int]] = []

    lite_model_predictions: List[List[int]] = []
    lite_model_labels: List[List[int]] = []
    lite_model_sizes: List[int] = []

    lite_model_no_quant_predictions: List[List[int]] = []
    lite_model_no_quant_labels: List[List[int]] = []
    lite_model_no_quant_sizes: List[int] = []
    label_names: [str] = []

    for i in range(training_count):
        print(f"Creating training and test sets. SNR: {snr}, Iteration: {i}")

        train_test_sets = sp_training.create_spectrum_painting_train_test_sets(
            spectrograms=spectrograms,
            label_names=classes,
            options=spectrum_painting_options,
            test_size=0.3
        )
        label_names = train_test_sets.label_names

        image_shape = train_test_sets.x_train_augmented[0].shape

        full_model = sp_model.create_tensorflow_model(image_shape=image_shape,
                                                      label_count=len(train_test_sets.label_names))

        sp_model.fit_model(full_model, train_test_sets, epochs=200, early_stopping_patience=20)
        print("\n")

        full_model_predictions.append([sp_predict.predict_full_model(full_model, x_a, x_p) for (x_a, x_p) in
                                       zip(train_test_sets.x_test_augmented, train_test_sets.x_test_painted)])
        full_model_labels.append(train_test_sets.y_test.astype(int).tolist())

        lite_model = sp_model.convert_to_tensorflow_lite(full_model,
                                                         train_test_sets.x_train_augmented[:100],
                                                         train_test_sets.x_train_painted[:100])

        lite_model_sizes.append(len(lite_model))

        lite_model_predictions.append([sp_predict.predict_lite_model(lite_model, x_a, x_p) for (x_a, x_p) in
                                       zip(train_test_sets.x_test_augmented, train_test_sets.x_test_painted)])
        lite_model_labels.append(train_test_sets.y_test.astype(int).tolist())

        print(f"Lite model accuracy = {calc_accuracy(lite_model_labels[-1], lite_model_predictions[-1])}")

        lite_no_quant_model = sp_model.convert_to_tensorflow_lite_no_quantization(full_model)

        lite_model_no_quant_sizes.append(len(lite_no_quant_model))

        lite_model_no_quant_predictions.append(
            [sp_predict.predict_lite_no_quant_model(lite_no_quant_model, x_a, x_p) for (x_a, x_p) in
             zip(train_test_sets.x_test_augmented, train_test_sets.x_test_painted)])

        lite_model_no_quant_labels.append(train_test_sets.y_test.astype(int).tolist())

    result = Result(
        snr=snr,
        label_names=label_names,
        full_model_labels=full_model_labels,
        full_model_predictions=full_model_predictions,
        lite_model_labels=lite_model_labels,
        lite_model_predictions=lite_model_predictions,
        lite_model_size=np.mean(lite_model_sizes),
        lite_model_no_quant_labels=lite_model_no_quant_labels,
        lite_model_no_quant_predictions=lite_model_no_quant_predictions,
        lite_no_quant_model_size=np.mean(lite_model_no_quant_sizes)
    )

    results.append(result.to_dict())

    print("Saving model")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(f"output/results-{timestamp}.json", "w") as f:
        json.dump({"results": results}, f)
