import tensorflow as tf

from xcenternet.model.backbone.upsample import upsample
from xcenternet.model.config import XModelMode
from xcenternet.model.layers import BatchNormalization
from xcenternet.model.preprocessing.preprocess_layer import PreprocessTFLayer


def create_mobilenetv2_10(height, width, pretrained: bool, mode: XModelMode = XModelMode.SIMPLE):
    shape = (height, width, 3)

    tf.keras.layers.BatchNormalization = BatchNormalization
    base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=shape, alpha=1.0, include_top=False, weights="imagenet" if pretrained else None)

    inputs = tf.keras.Input(shape=shape, name="input")
    x = tf.cast(inputs, tf.float32)/255.0

    base_model, features = upsample(
        base_model,
        x,
        ["block_3_expand_relu", "block_6_expand_relu", "block_13_expand_relu", "out_relu"],  # 96, 96, 288, 1280
        mode,
    )
    return base_model, inputs, features


def create_mobilenetv2_035(height, width, pretrained: bool, mode: XModelMode = XModelMode.SIMPLE):
    shape = (height, width, 3)

    tf.keras.layers.BatchNormalization = BatchNormalization
    base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=shape, alpha=0.35, include_top=False, weights="imagenet" if pretrained else None)

    inputs = tf.keras.Input(shape=shape, name="input")
    x = tf.cast(inputs, tf.float32)/255.0

    base_model, features = upsample(
        base_model,
        x,
        ["block_3_expand_relu", "block_6_expand_relu", "block_13_expand_relu", "out_relu"],  # 48, 96, 192, 1280
        mode,
    )
    return base_model, inputs, features
