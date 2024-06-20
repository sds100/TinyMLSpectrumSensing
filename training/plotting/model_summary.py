import tensorflow as tf

model = tf.keras.models.load_model("../output/spectrum-painting-model-filters-2.keras")
model.summary()
