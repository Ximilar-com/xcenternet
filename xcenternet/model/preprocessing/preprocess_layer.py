import numpy as np
import tensorflow as tf


@tf.function
def preprocess_tf(x):
    """
    Preprocessing for Keras (MobileNetV2, ResNetV2).
    :param x: np.asarray([image, image, ...], dtype="float32") in RGB
    :return: normalized image tf style (RGB)
    """
    batch, height, width, channels = x.shape
    x = tf.cast(x, tf.float32)

    # ! do not use tf.constant as they are not right now serializable when saving model for .h5
    # ! https://stackoverflow.com/questions/47066635/checkpointing-keras-model-typeerror-cant-pickle-thread-lock-objects
    # mean_tensor = tf.constant([127.5, 127.5, 127.5], dtype=tf.float32, shape=[1, 1, 1, 3], name="mean")
    # one_tensor = tf.constant([1.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 1, 3], name="one")

    mean_tensor = np.asarray([[[[127.5, 127.5, 127.5]]]], dtype=np.float32)
    one_tensor = np.asarray([[[[1.0, 1.0, 1.0]]]], dtype=np.float32)

    x = tf.keras.backend.reshape(x, (-1, 3))
    result = (x / mean_tensor) - one_tensor
    return tf.keras.backend.reshape(result, (-1, height, width, channels))


@tf.function
def preprocess_caffe(x):
    """
    Preprocessing for Keras (VGG, ResnetV1).
    ! This works only for channels_last
    :param x: np.asarray([image, image, ...], dtype="float32") in RGB
    :return: normalized image vgg style (BGR)
    """
    batch, height, width, channels = x.shape
    x = tf.cast(x, tf.float32)
    r, g, b = tf.split(x, 3, axis=3)
    x = tf.concat([b, g, r], 3)
    mean_tensor = np.asarray([[[[103.939, 116.779, 123.68]]]], dtype=np.float32)
    result = x - mean_tensor
    return tf.keras.backend.reshape(result, (-1, height, width, channels))


class PreprocessCaffeLayer(tf.keras.layers.Layer):
    def __init__(self, name="preprocess_cafe", **kwargs):
        super(PreprocessCaffeLayer, self).__init__(name=name, **kwargs)
        self.preprocess = preprocess_caffe

    def call(self, input, **kwargs):
        return self.preprocess(input)

    def get_config(self):
        config = super(PreprocessCaffeLayer, self).get_config()
        return config


class PreprocessTFLayer(tf.keras.layers.Layer):
    def __init__(self, name="preprocess_tf", **kwargs):
        super(PreprocessTFLayer, self).__init__(name=name, **kwargs)
        self.preprocess = preprocess_tf

    def call(self, input, **kwargs):
        return self.preprocess(input)

    def get_config(self):
        config = super(PreprocessTFLayer, self).get_config()
        return config
