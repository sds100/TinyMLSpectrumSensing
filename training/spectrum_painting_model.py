from typing import List

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from tensorflow.keras import models


def predict_full_model(model: models.Model,
                       x_augmented: npt.NDArray[np.uint8],
                       x_painted: npt.NDArray[np.uint8]) -> int:
    x_augmented.shape += (1,)
    x_augmented = (np.expand_dims(x_augmented, 0))

    x_painted.shape += (1,)
    x_painted = (np.expand_dims(x_painted, 0))

    predictions_single = model.predict(x=[(x_augmented, x_painted)])
    prediction_index = np.argmax(predictions_single[0])
    return prediction_index


def predict_lite_model(model: List[bytes],
                       x_augmented: npt.NDArray[np.uint8],
                       x_painted: npt.NDArray[np.uint8]) -> int:
    x_augmented.shape += (1,)
    x_augmented = (np.expand_dims(x_augmented, 0))

    x_painted.shape += (1,)
    x_painted = (np.expand_dims(x_painted, 0))

    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    interpreter.set_tensor(0, x_augmented)
    interpreter.set_tensor(1, x_painted)

    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details["index"])[0]
    prediction_index = np.argmax(prediction)
    return prediction_index
