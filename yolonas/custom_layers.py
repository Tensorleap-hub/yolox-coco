import tensorflow as tf
from tensorflow import TensorShape


class MockOneClass(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MockOneClass, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return tf.reduce_max(inputs, axis=1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return TensorShape(1, 1, 8400)
