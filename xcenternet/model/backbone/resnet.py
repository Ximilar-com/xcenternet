import tensorflow as tf
from classification_models.tfkeras import Classifiers

from xcenternet.model.backbone.upsample import upsample
from xcenternet.model.config import XModelMode, XModelType
from xcenternet.model.constants import L2_REG
from xcenternet.model.layers import BatchNormalization, set_regularization
from xcenternet.model.preprocessing.preprocess_layer import PreprocessCaffeLayer


def create_resnet(height, width, pretrained: bool, mode: XModelMode = XModelMode.SIMPLE, mtype: XModelType = XModelType.CENTERNET):
    shape = (height, width, 3)

    tf.keras.layers.BatchNormalization = BatchNormalization
    base_model = tf.keras.applications.resnet50.ResNet50(
        input_shape=shape, include_top=False, weights="imagenet" if pretrained else None, layers=tf.keras.layers
    )

    inputs = tf.keras.Input(shape=shape, name="input")
    x = PreprocessCaffeLayer()(inputs)

    base_model, features = upsample(
        base_model, x, ["conv2_block3_out", "conv3_block3_out", "conv4_block3_out", "conv5_block3_out"], mode, mtype
    )

    return base_model, inputs, features


def create_resnet_18(height, width, pretrained: bool, mode: XModelMode = XModelMode.SIMPLE, mtype: XModelType = XModelType.CENTERNET):
    shape = (height, width, 3)

    ResNet18, preprocess_input = Classifiers.get("resnet18")
    weights = "imagenet" if pretrained else None
    base_model = ResNet18(input_shape=shape, weights=weights, include_top=False)
    base_model = set_regularization(base_model, kernel_regularizer=tf.keras.regularizers.l2(L2_REG))

    inputs = tf.keras.Input(shape=shape, name="input")
    base_model, features = upsample(
        base_model, inputs, ["stage2_unit1_relu1", "stage3_unit1_relu1", "stage4_unit1_relu1", "relu1"], mode, mtype
    )

    return base_model, inputs, features
