import tensorflow as tf
from tensorflow import TensorShape


class MockNClass(tf.keras.layers.Layer):
    def __init__(self, n: int, **kwargs):
        super(MockNClass, self).__init__(**kwargs)
        self.n_classes = n

    def call(self, inputs, **kwargs):
        return inputs[:, :self.n_classes, :]

    def compute_output_shape(self, input_shape):
        return TensorShape(1, self.n_classes, 8400)
