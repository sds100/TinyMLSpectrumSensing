from typing import List

import keras.callbacks
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from tensorflow.keras import models, layers, losses

from training.spectrum_painting_training import SpectrumPaintingTrainTestSets


def create_channel(input: layers.Input) -> layers.Layer:
    # Padding "same" adds zero-padding.
    layer = layers.Conv2D(filters=2, kernel_size=(7, 7), activation='relu', padding='same')(input)
    layer = layers.BatchNormalization()(layer)
    layer = layers.MaxPooling2D((2, 2))(layer)

    layer = layers.Conv2D(filters=2, kernel_size=(5, 5), activation='relu', padding='same')(layer)
    layer = layers.BatchNormalization()(layer)
    layer = layers.MaxPooling2D((2, 2))(layer)

    layer = layers.Conv2D(filters=2, kernel_size=(3, 3), activation='relu', padding='same')(layer)
    layer = layers.BatchNormalization()(layer)
    layer = layers.MaxPooling2D((2, 2))(layer)

    return layer


def create_tensorflow_model(image_shape: (int, int), label_count: int) -> models.Model:
    # The input shape to the CNN is the height, width and number of color channels. The spectrograms
    # only have one color channel.
    input_shape = (image_shape[0], image_shape[1], 1)

    augmented_input = layers.Input(shape=input_shape)
    augmented_channel = create_channel(augmented_input)

    painted_input = layers.Input(shape=input_shape)
    painted_channel = create_channel(painted_input)

    output = layers.Concatenate()([augmented_channel, painted_channel])

    # Flatten the 3D image output to 1 dimension
    output = layers.Flatten()(output)

    output = layers.Dense(label_count)(output)

    tf_model = models.Model(inputs=[augmented_input, painted_input], outputs=[output])

    return tf_model


def create_tensorflow_model_one_channel(image_shape: (int, int), label_count: int) -> models.Model:
    # The input shape to the CNN is the height, width and number of color channels. The spectrograms
    # only have one color channel.
    input_shape = (image_shape[0], image_shape[1], 1)

    input_layer = layers.Input(shape=input_shape)
    channel = create_channel(input_layer)

    # Flatten the 3D image output to 1 dimension
    output = layers.Flatten()(channel)

    output = layers.Dense(label_count)(output)

    tf_model = models.Model(inputs=[input_layer], outputs=[output])

    return tf_model


def fit_model(model: models.Model,
              train_test_sets: SpectrumPaintingTrainTestSets,
              epochs: int,
              early_stopping_patience: int):
    model.compile(optimizer='adam',
                  loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    class CustomCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # print the epoch and accuracy on the same line. The carriage return and empty end character
            # are required to do this.
            print("\r", f"Epoch: {epoch}, Val. accuracy = {logs.get('val_accuracy')}", end="")

    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=early_stopping_patience,
                                                            min_delta=0.02)
    # convert ints to the type of int that can be used in a Tensor
    history = model.fit(x=[train_test_sets.x_train_augmented, train_test_sets.x_train_painted],
                        y=train_test_sets.y_train,
                        epochs=epochs,
                        validation_data=(
                            [train_test_sets.x_test_augmented, train_test_sets.x_test_painted],
                            train_test_sets.y_test),
                        verbose=0,
                        callbacks=[CustomCallback(), early_stopping_callback], )

    return history


def fit_model_one_channel(model: models.Model,
                          train_test_sets: SpectrumPaintingTrainTestSets,
                          epochs: int,
                          early_stopping_patience: int):
    model.compile(optimizer='adam',
                  loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    class CustomCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # print the epoch and accuracy on the same line. The carriage return and empty end character
            # are required to do this.
            print("\r", f"Epoch: {epoch}, Val. accuracy = {logs.get('val_accuracy')}", end="")

    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=early_stopping_patience)
    # convert ints to the type of int that can be used in a Tensor
    history = model.fit(x=[train_test_sets.x_train_augmented],
                        y=train_test_sets.y_train,
                        epochs=epochs,
                        validation_data=(
                            [train_test_sets.x_test_augmented],
                            train_test_sets.y_test),
                        verbose=0,
                        callbacks=[CustomCallback(), early_stopping_callback], )

    return history


def convert_to_tensorflow_lite(model: models.Model,
                               augmented_test_images: List[npt.NDArray[np.uint8]],
                               painted_test_images: List[npt.NDArray[np.uint8]]):
    """
    Convert the full tensorflow model to a Lite model.
    """

    def representative_data_gen():
        """
        Convert test images to float32 and the correct dimensions
        for TensorFlow to do full-integer quantization.
        """
        repr_augmented_images = np.copy(np.asarray(augmented_test_images))
        repr_augmented_images = [img.astype(np.float32) for img in repr_augmented_images]

        for img in repr_augmented_images:
            img.shape += (1,)

        repr_painted_images = np.copy(np.asarray(painted_test_images))
        repr_painted_images = [img.astype(np.float32) for img in repr_painted_images]

        for img in repr_painted_images:
            img.shape += (1,)

        np.random.shuffle(repr_augmented_images)
        np.random.shuffle(repr_painted_images)

        repr_augmented_images = (np.expand_dims(repr_augmented_images, 0))
        repr_painted_images = (np.expand_dims(repr_painted_images, 0))

        for aug_value, painted_value in list(zip(repr_augmented_images, repr_painted_images)):
            # Model has only one input so each data point has one element.
            yield [aug_value, painted_value]

    # This requires TensorFlow <= 2.15.0 for it to work. See https://github.com/tensorflow/tensorflow/issues/63987
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    return converter.convert()


def convert_to_tensorflow_lite_one_channel(model: models.Model,
                                           test_images: List[npt.NDArray[np.uint8]]):
    """
    Convert the full tensorflow model to a Lite model.
    """

    def representative_data_gen():
        """
        Convert test images to float32 and the correct dimensions
        for TensorFlow to do full-integer quantization.
        """
        repr_test_images = np.copy(test_images)
        repr_test_images = [img.astype(np.float32) for img in repr_test_images]

        for img in repr_test_images:
            img.shape += (1,)

        images = tf.data.Dataset.from_tensor_slices(repr_test_images).batch(1).take(100)
        for img in images:
            # Model has only one input so each data point has one element.
            yield [img]

    # This requires TensorFlow <= 2.15.0 for it to work. See https://github.com/tensorflow/tensorflow/issues/63987
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    return converter.convert()
