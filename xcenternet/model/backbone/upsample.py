import tensorflow as tf

from xcenternet.model.config import XModelMode
from xcenternet.model.layers import (
    deConv2DBatchNorm,
    conv2DBatchNorm,
    upsampleConcat,
    upsampleSum,
    deformConv2D,
    deformConv2DShortcut,
    coord_conv,
)


def upsample(base_model, x, layers, mode: XModelMode):
    """
    Feature extraction upsampling/generation before regression and heatmap heads.
    :param base_model: tf.keras.Model object
    :param x: output from last layer of base_model 
    :param layers: layer names from backbone
    :param mode: upsampling mode for features
    :return: base_model and features
    """
    with tf.name_scope("upsample"):
        layers = [base_model.get_layer(layer_name) for layer_name in layers]
        model_multi_output = tf.keras.Model(
            inputs=base_model.input, outputs=[layer.output for layer in layers], name="multioutput"
        )
        c2, c3, c4, c5 = model_multi_output(x)

        c5 = tf.keras.layers.Dropout(rate=0.5)(c5)
        c4 = tf.keras.layers.Dropout(rate=0.4)(c4)
        c3 = tf.keras.layers.Dropout(rate=0.3)(c3)
        c2 = tf.keras.layers.Dropout(rate=0.2)(c2)

        c5 = coord_conv(c5, with_r=True)

        # You can create your own upsample layer for example FPN
        # if you need here, you need to also add this mode to the XmodelMode enum

        if mode == XModelMode.SIMPLE:
            deconv = deConv2DBatchNorm(c5, filters=256, kernel_size=(4, 4), name="deconv1")
            deconv = deConv2DBatchNorm(deconv, filters=128, kernel_size=(4, 4), name="deconv2")
            features = deConv2DBatchNorm(deconv, filters=64, kernel_size=(4, 4), name="features")
            return base_model, features

        if mode == XModelMode.DCN:
            dcn = deformConv2D(c5, 256, name="up1")
            dcn = deformConv2D(dcn, 128, name="up2")
            features = deformConv2D(dcn, 64, name="features")
            return base_model, features

        if mode == XModelMode.DCNSHORTCUT or mode == XModelMode.DCNSHORTCUTCONCAT:
            ratio = 2 if mode == XModelMode.DCNSHORTCUTCONCAT else 1
            dcn = deformConv2DShortcut(c5, c4, 1, filters=256 / ratio, mode=mode, name="deformconv1")
            dcn = deformConv2DShortcut(dcn, c3, 2, filters=128 / ratio, mode=mode, name="deformconv2")
            features = deformConv2DShortcut(dcn, c2, 3, filters=64 / ratio, mode=mode, name="features")
            return base_model, features

        if mode == XModelMode.CONCAT:
            x = upsampleConcat(256, 256, c5, c4, name="up1")
            x = upsampleConcat(128, 128, x, c3, name="up2")
            features = upsampleConcat(64, 64, x, c2, name="features")
            return base_model, features

        if mode == XModelMode.SUM:
            p5 = conv2DBatchNorm(c5, filters=128, kernel_size=(1, 1), name="conv2D_up5")
            p4 = upsampleSum(p5, c4, name="up4")
            p3 = upsampleSum(p4, c3, name="up3")
            p2 = upsampleSum(p3, c2, name="up2")
            features = conv2DBatchNorm(p2, filters=128, kernel_size=(3, 3), name="features")
            return base_model, features

        raise ValueError(f"Unsupported mode {mode}.")
