import numpy as np
import tensorflow as tf

from xcenternet.model.backbone.efficientnet import create_efficientnetb0
from xcenternet.model.backbone.resnet import create_resnet, create_resnet_18
from xcenternet.model.centernet import XCenternetModel, XTTFModel
from xcenternet.model.config import XModelType, XModelBackbone, XModelMode
from xcenternet.model.constants import L2_REG, ACTIVATION, KERNEL_INIT
from xcenternet.model.layers import BatchNormalization

CREATE_MODELS = {
    XModelBackbone.RESNET18: lambda w, h, pretrained, mode: create_resnet_18(w, h, pretrained, mode),
    XModelBackbone.RESNET50: lambda w, h, pretrained, mode: create_resnet(w, h, pretrained, mode),
    XModelBackbone.EFFICIENTNETB0: lambda w, h, pretrained, mode: create_efficientnetb0(w, h, pretrained, mode),
}


def create_model(
    image_size,
    labels,
    model_type=XModelType.CENTERNET,
    backbone=XModelBackbone.EFFICIENTNETB0,
    mode=XModelMode.CONCAT,
    pretrained_backbone=True,
):
    """
    Creates a new TensorFlow model.

    :param image_size: image height and width
    :param labels: number of labels
    :param backbone: backbone to be used for creating a new model (pre-trained if available)
    :return: new model (XCenternetModel)
    """
    input, features = _create_backbone(image_size, pretrained_backbone, backbone=backbone, mode=mode)
    return _finish_model(labels, input, features, model_type)


def load_and_update_model(model_dir: str, labels: int, model_type: XModelType, feature_layer="features"):
    """
    Loads model from given directory and update it to the new number of labels.

    :param model_dir: directory with model in TensorFlow SavedModel format
    :param labels: number of labels in the new model
    :param model_type:
    :return: loadel model (XCenternetModel)
    """
    input, features = _load_backbone(model_dir, feature_layer=feature_layer)
    return _finish_model(labels, input, features, model_type)


def load_pretrained_weights(model, weights_path, reset_heads=True):
    """
    Loads pretrained weights for given model by name. By default, the heads are reset to default values.
    The heads in a new model might have a same shape as in the pretrained one. But we should not keep them
    and instead train the from scratch.

    :param model: Non-trained model.
    :param weights_path: Path to file with pretrained model weights.
    :param reset_heads: reset weights for heatmap, size and offset of present
    :return: None
    """

    def load():
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    if not reset_heads:
        load()
        return

    # I did not find a way hot to reinitialize weights with proper initializer after they have been changed.
    # Remembering the weights from an untrained model and setting them after loading pretrained weights
    # will do the trick.
    layers_to_reset = _layers_to_reset(model)
    init_weights = {l.name: l.get_weights() for l in layers_to_reset}

    load()

    for layer in model.layers:
        if layer in init_weights:
            layer.set_weights(init_weights[layer])


def _layers_to_reset(model):
    heads_start_names = ["heatmap_conv2D", "size_conv2D", "offset_conv2D"]

    reinitialize = False
    result = []
    for layer in model.layers:
        reinitialize = reinitialize or layer.name in heads_start_names
        if reinitialize:
            result.append(layer)

    return result


def _create_backbone(image_size, pretrained: bool, backbone: XModelBackbone, mode: XModelMode):
    # get backbone model
    if backbone not in CREATE_MODELS:
        raise Exception(f"Model {backbone} does not exist!")

    _, input, features = CREATE_MODELS[backbone](image_size, image_size, pretrained, mode)
    return input, features


def _load_backbone(model_path, feature_layer="features"):
    model = tf.keras.models.load_model(model_path)
    input = model.input
    features = model.get_layer(feature_layer).output

    return input, features


def _finish_model(labels: int, input, features, model_type: XModelType):
    outputs = []

    # output layers
    with tf.name_scope("heatmap"):
        output_heatmap = tf.keras.layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            use_bias=False,
            kernel_initializer=tf.keras.initializers.RandomNormal(0.01),
            kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
            name="heatmap_conv2D",
        )(features)
        output_heatmap = BatchNormalization(name="heatmap_norm")(output_heatmap)
        output_heatmap = tf.keras.layers.Activation(ACTIVATION, name="heatmap_activ")(output_heatmap)
        output_heatmap = tf.keras.layers.Conv2D(
            labels,
            (1, 1),
            padding="valid",
            activation=tf.nn.sigmoid,
            kernel_initializer=tf.keras.initializers.RandomNormal(0.01),
            kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
            bias_initializer=tf.constant_initializer(-np.log((1.0 - 0.1) / 0.1)),
            name="heatmap",
        )(output_heatmap)
        outputs.append(output_heatmap)

    with tf.name_scope("bbox_size"):
        reg_size = 2 if model_type == XModelType.CENTERNET else 4
        output_bbox_size = tf.keras.layers.Conv2D(
            64,
            (3, 3),
            padding="same",
            use_bias=False,
            kernel_initializer=tf.keras.initializers.RandomNormal(0.001),
            kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
            name="size_conv2D",
        )(features)
        output_bbox_size = BatchNormalization(name="size_norm")(output_bbox_size)
        output_bbox_size = tf.keras.layers.Activation(ACTIVATION, name="size_activ")(output_bbox_size)
        output_bbox_size = tf.keras.layers.Conv2D(
            reg_size,
            (1, 1),
            padding="valid",
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomNormal(0.001),
            kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
            name="bounding_box_size",
        )(output_bbox_size)
        if model_type != XModelType.CENTERNET:
            output_bbox_size = 16.0 * tf.keras.layers.Activation(ACTIVATION, name="size_activ2")(output_bbox_size)
        outputs.append(output_bbox_size)

    if model_type == XModelType.CENTERNET:
        with tf.name_scope("local_offset"):
            output_local_offset = tf.keras.layers.Conv2D(
                64,
                [3, 3],
                padding="same",
                use_bias=False,
                kernel_initializer=KERNEL_INIT,
                kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
                name="offset_conv2D",
            )(features)
            output_local_offset = BatchNormalization(name="offset_norm")(output_local_offset)
            output_local_offset = tf.keras.layers.Activation(ACTIVATION, name="offset_activ")(output_local_offset)
            output_local_offset = tf.keras.layers.Conv2D(
                2,
                (1, 1),
                padding="valid",
                activation=None,
                kernel_initializer=KERNEL_INIT,
                kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
                name="local_offset",
            )(output_local_offset)
        outputs.append(output_local_offset)
        return XCenternetModel(inputs=input, outputs=outputs, name=model_type.name.lower())
    return XTTFModel(inputs=input, outputs=outputs, name=model_type.name.lower())
