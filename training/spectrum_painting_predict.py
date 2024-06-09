from typing import List

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from tensorflow.keras import models


def predict_full_model(model: models.Model,
                       x_augmented: npt.NDArray[np.uint8],
                       x_painted: npt.NDArray[np.uint8]) -> int:
    x_augmented_copy = np.copy(x_augmented)
    x_painted_copy = np.copy(x_painted)

    x_augmented_copy.shape += (1,)
    x_augmented_copy = (np.expand_dims(x_augmented_copy, 0))

    x_painted_copy.shape += (1,)
    x_painted_copy = (np.expand_dims(x_painted_copy, 0))

    predictions_single = model.predict(x=[(x_augmented_copy, x_painted_copy)], verbose=0)
    prediction_index = np.argmax(predictions_single[0])
    return prediction_index


def predict_full_model_one_channel(model: models.Model, x_test: npt.NDArray[np.uint8]) -> int:
    x_test_copy = np.copy(x_test)

    x_test_copy.shape += (1,)
    x_test_copy = (np.expand_dims(x_test_copy, 0))

    predictions_single = model.predict(x=[x_test_copy], verbose=0)
    prediction_index = np.argmax(predictions_single[0])
    return prediction_index


def predict_lite_model(model: List[bytes],
                       x_augmented: npt.NDArray[np.uint8],
                       x_painted: npt.NDArray[np.uint8]) -> int:
    x_augmented.shape += (1,)
    x_augmented = (np.expand_dims(x_augmented, 0))
    x_augmented = x_augmented.astype(np.uint8)

    x_painted.shape += (1,)
    x_painted = (np.expand_dims(x_painted, 0))
    x_painted = x_painted.astype(np.uint8)

    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    print(input_details)
    output_details = interpreter.get_output_details()[0]

    interpreter.set_tensor(0, x_augmented)
    interpreter.set_tensor(1, x_painted)

    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details["index"])[0]
    prediction_index = np.argmax(prediction)
    return prediction_index


def predict_lite_model_one_channel(model: List[bytes], test_image: npt.NDArray[np.uint8]) -> int:
    test_image.shape += (1,)
    test_image = (np.expand_dims(test_image, 0))
    test_image = test_image.astype(np.uint8)

    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    interpreter.set_tensor(0, test_image)

    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details["index"])[0]
    prediction_index = np.argmax(prediction)
    return prediction_index
