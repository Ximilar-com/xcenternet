import tensorflow as tf
import tensorflow.python.keras.applications.efficientnet as efficientnet

from xcenternet.model.backbone.upsample import upsample
from xcenternet.model.config import XModelMode
from xcenternet.model.layers import BatchNormalization


def load_weights(weights, model_name="efficientnetb0", include_top=False):
    if weights == "imagenet":
        if include_top:
            file_suffix = ".h5"
            file_hash = efficientnet.WEIGHTS_HASHES[model_name[-2:]][0]
        else:
            file_suffix = "_notop.h5"
            file_hash = efficientnet.WEIGHTS_HASHES[model_name[-2:]][1]
        file_name = model_name + file_suffix
        weights_path = efficientnet.data_utils.get_file(
            file_name, efficientnet.BASE_WEIGHTS_PATH + file_name, cache_subdir="models", file_hash=file_hash
        )
    return weights_path


def create_efficientnetb0(height, width, pretrained: bool, mode: XModelMode = XModelMode.SIMPLE):
    shape = (height, width, 3)

    efficientnet.layers.BatchNormalization = BatchNormalization
    base_model = efficientnet.EfficientNetB0(input_shape=shape, include_top=False, weights=None)

    if pretrained:
        base_model.load_weights(load_weights("imagenet", model_name="efficientnetb0"), by_name=True, skip_mismatch=True)
        print("\033[31m", "Imagenet model loaded", "\033[0m")
    else:
        print("\033[31m", "Imagenet model not loaded", "\033[0m")

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

    efficientnet.layers.BatchNormalization = BatchNormalization
    base_model = efficientnet.EfficientNetB1(input_shape=shape, include_top=False, weights=None)

    if pretrained:
        base_model.load_weights(load_weights("imagenet", model_name="efficientnetb1"), by_name=True, skip_mismatch=True)

    inputs = tf.keras.Input(shape=shape, name="input")
    base_model, features = upsample(
        base_model, inputs, ["block2c_add", "block3c_add", "block5d_add", "block6e_add"], mode
    )
    return base_model, inputs, features


def create_efficientnetb2(height, width, pretrained: bool, mode: XModelMode = XModelMode.SIMPLE):
    shape = (height, width, 3)
    base_model = efficientnet.EfficientNetB2(input_shape=shape, include_top=False, weights=None)

    if pretrained:
        base_model.load_weights(load_weights("imagenet", model_name="efficientnetb2"), by_name=True, skip_mismatch=True)

    inputs = tf.keras.Input(shape=shape, name="input")
    base_model, features = upsample(
        base_model, inputs, ["block2c_add", "block3c_add", "block5d_add", "block6e_add"], mode
    )
    return base_model, inputs, features
