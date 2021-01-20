import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2

from xcenternet.model.backbone.upsample import upsample
from xcenternet.model.config import XModelMode
from xcenternet.model.layers import BatchNormalization


def create_efficientnetb0(height, width, pretrained: bool, mode: XModelMode = XModelMode.SIMPLE):
    shape = (height, width, 3)

    tf.keras.layers.BatchNormalization = BatchNormalization
    base_model = EfficientNetB0(input_shape=shape, include_top=False, weights="imagenet" if pretrained else None)

    inputs = tf.keras.Input(shape=shape, name="input")
    base_model, features = upsample(
        base_model,
        inputs,
        ["block2b_activation", "block3b_activation", "block5c_activation", "top_activation"],  # 144, 240, 672, 1280
        # ["block2b_add", "block3b_add", "block5c_add", "block6d_add"], # 24, 40, 112, 192
        # resnet50: 256, 512, 1024, 2048
        mode,
    )
    return base_model, inputs, features


def create_efficientnetb1(height, width, pretrained: bool, mode: XModelMode = XModelMode.SIMPLE):
    shape = (height, width, 3)

    tf.keras.layers.BatchNormalization = BatchNormalization
    base_model = EfficientNetB1(input_shape=shape, include_top=False, weights="imagenet" if pretrained else None)

    inputs = tf.keras.Input(shape=shape, name="input")
    base_model, features = upsample(
        base_model, inputs, ["block2c_add", "block3c_add", "block5d_add", "block6e_add"], mode
    )
    return base_model, inputs, features


def create_efficientnetb2(height, width, pretrained: bool, mode: XModelMode = XModelMode.SIMPLE):
    shape = (height, width, 3)

    tf.keras.layers.BatchNormalization = BatchNormalization
    base_model = EfficientNetB2(input_shape=shape, include_top=False, weights="imagenet" if pretrained else None)

    inputs = tf.keras.Input(shape=shape, name="input")
    base_model, features = upsample(
        base_model, inputs, ["block2c_add", "block3c_add", "block5d_add", "block6e_add"], mode
    )
    return base_model, inputs, features
