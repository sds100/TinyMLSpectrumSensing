import time

import tensorflow as tf

import spectrum_painting_data as sp_data
import spectrum_painting_model as sp_model
import spectrum_painting_predict as sp_predict
import spectrum_painting_training as sp_training

full_model = tf.keras.models.load_model('output/spectrum-painting-model-batch-norm-filters-8-iteration-1.keras')

classes = ["Z", "B", "W", "BW", "ZB", "ZW", "ZBW"]
snr_list = [0, 5, 10, 15, 20, 25, 30]

spectrograms = sp_data.load_spectrograms(data_dir="data/numpy",
                                         classes=classes,
                                         snr_list=snr_list,
                                         windows_per_spectrogram=256,
                                         window_length=256,
                                         nfft=64,
                                         spectrogram_count=10)

spectrum_painting_options = sp_training.SpectrumPaintingTrainingOptions(
    downsample_resolution=64,
    k=3,
    l=16,
    d=4
)

train_test_sets = sp_training.create_spectrum_painting_train_test_sets(
    spectrograms=spectrograms,
    label_names=classes,
    options=spectrum_painting_options,
    test_size=0.3
)

lite_model = sp_model.convert_to_tensorflow_lite(full_model,
                                                 train_test_sets.x_test_augmented,
                                                 train_test_sets.x_test_painted)
print(len(lite_model))

with open("output/spectrum-painting-model-quant-filters-1.tflite", "wb") as f:
    f.write(lite_model)

no_quantization_model = sp_model.convert_to_tensorflow_lite_no_quantization(full_model)

with open("output/spectrum-painting-model-optimize-filters-8.tflite", "wb") as f:
    f.write(no_quantization_model)

print(len(no_quantization_model))

start = time.thread_time()
prediction = sp_predict.predict_lite_no_quant_model(no_quantization_model,
                                                    x_augmented=train_test_sets.x_test_augmented[0],
                                                    x_painted=train_test_sets.x_test_painted[0])
end = time.thread_time()

print("Old = 0.002608")
print(end - start)
