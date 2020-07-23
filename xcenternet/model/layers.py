import tensorflow as tf
import tensorflow_addons as tfa

from xcenternet.model.config import XModelMode
from xcenternet.model.constants import BN_MOMENTUM, L2_REG, ACTIVATION, KERNEL_INIT


def conv2DBatchNorm(inputs, filters=128, kernel_size=(2, 2), activation=ACTIVATION, name=None):
    """
    Convolution 2D with BatchNormalization and activation
    :param inputs: input to the layer
    :param filters: number of filters from convoltuion, defaults to 128
    :param kernel_size: kernel for convolution, defaults to (2, 2)
    :param name: name of the scope
    :return: output from convolution layer
    """
    with tf.name_scope(name) as scope:
        layer = tf.keras.layers.Conv2D(
            filters, kernel_size, padding="same", kernel_initializer=KERNEL_INIT, use_bias=False
        )(inputs)
        layer = BatchNormalization()(layer)
        layer = tf.keras.layers.Activation(activation, name=name)(layer)
        return layer


def deformConv2D(inputs, filters=64, kernel_size=(4, 4), strides=(2, 2), activation=ACTIVATION, name=None):
    """
    Apply deformable with conv2d transpose on inputs.
    :param inputs: input from upper layer
    :param filters: numbere of filters, defaults to 64
    :param kernel_size: kernel size, defaults to (4, 4)
    :param strides: kernel strides, defaults to (2, 2)
    :param activation: relu activation, defaults to ACTIVATION
    :param name: name of the scope, defaults to None
    :return: output from sequence of layers
    """
    from tensorflow_addons.layers.deformable_conv2d import DeformableConv2D

    with tf.name_scope(name) as scope:
        fc = DeformableConv2D(
            filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            use_bias=False,
            padding="same",
            kernel_initializer=KERNEL_INIT,
            kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        )(inputs)
        layer = BatchNormalization()(fc)
        layer = tf.keras.layers.Activation(activation)(layer)
        up = tf.keras.layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides=strides,
            padding="same",
            use_bias=False,
            kernel_initializer=KERNEL_INIT,
            kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        )(layer)
        layer = BatchNormalization()(up)
        layer = tf.keras.layers.Activation(activation)(layer)
        return layer


def deformConv2DShortcut(
    inputs,
    shortcut,
    shortcut_number,
    filters=64,
    kernel_size=(3, 3),
    strides=(1, 1),
    activation=ACTIVATION,
    mode=None,
    name=None,
):
    """
    Similar to TTF Net, we are merging deformable conv2d with shortcut connections from backbone.
    :param inputs: input to the deformable convolution
    :param shortcut: output from backbone
    :param shortcut_number: how many layers we should apply before merging shortuct with deformable convolution
    :param filters: number of filters to compute, defaults to 64
    :param kernel_size: conv kernel size, defaults to (3, 3)
    :param strides: conv strides, defaults to (1, 1)
    :param activation: name of activation layer, 
    :param mode: concat or sum, defaults to "sum" (ttfnet)
    :param name: name of the scope, defaults to None
    :return: output from sequence of layers
    """
    from tensorflow_addons.layers.deformable_conv2d import DeformableConv2D

    filters = int(filters)

    with tf.name_scope(name) as scope:
        dcn = DeformableConv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            use_bias=False,
            kernel_initializer=KERNEL_INIT,
            kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        )(inputs)
        dcn = BatchNormalization()(dcn)
        dcn = tf.keras.layers.Activation(activation)(dcn)
        dcn = tf.keras.layers.UpSampling2D(interpolation="bilinear")(dcn)

        # shortcut from backbone
        for i in range(shortcut_number):
            bias = False if i < (shortcut_number - 1) or mode == XModelMode.DCNSHORTCUTCONCAT else True
            shortcut = tf.keras.layers.Conv2D(
                filters,
                kernel_size=kernel_size,
                strides=strides,
                padding="same",
                use_bias=bias,
                kernel_initializer=KERNEL_INIT,
                kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
            )(shortcut)

            # do not call activation layer on the last output from shortcut
            if i < (shortcut_number - 1) or mode == XModelMode.DCNSHORTCUTCONCAT:
                shortcut = BatchNormalization()(shortcut)
                shortcut = tf.keras.layers.Activation(activation)(shortcut)

        if mode == XModelMode.DCNSHORTCUTCONCAT:
            return tf.keras.layers.Concatenate()([dcn, shortcut])

        return dcn + shortcut


def deConv2DBatchNorm(inputs, filters=128, kernel_size=(2, 2), strides=(2, 2), activation=ACTIVATION, name=None):
    """
    Deconvolution with BatchNormalization and activation
    :param inputs: input to the layer
    :param filters: number of filters from convoltuion, defaults to 128
    :param kernel_size: kernel for convolution, defaults to (2, 2)
    :param strides: strides of convolution, defaults to (2, 2)
    :param name: name of the scope
    :return: output deconvolution layer
    """
    with tf.name_scope(name) as scope:
        layer = tf.keras.layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides=strides,
            padding="same",
            use_bias=False,
            kernel_initializer=KERNEL_INIT,
            kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        )(inputs)
        layer = BatchNormalization()(layer)
        layer = tf.keras.layers.Activation(activation)(layer)
        return layer


def upsampleConcat(filter_out1, filter_out2, z, p, activation=ACTIVATION, name=None):
    """
    Upsample output from convolution and merge it with another convolution
    :param filter_out1: number of upsampling filter
    :param filter_out2: number of output filter
    :param z: input to upsample
    :param p: input to merge with upsample
    :param name: name of the scope
    :return: upsampled output
    """
    with tf.name_scope(name) as scope:
        x = tf.keras.layers.UpSampling2D()(z)
        x = tf.keras.layers.Conv2D(
            filter_out1,
            1,
            padding="same",
            use_bias=False,
            kernel_initializer=KERNEL_INIT,
            kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        )(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
        x = tf.keras.layers.Concatenate()([p, x])
        x = tf.keras.layers.Conv2D(
            filter_out2,
            3,
            padding="same",
            use_bias=False,
            kernel_initializer=KERNEL_INIT,
            kernel_regularizer=tf.keras.regularizers.l2(L2_REG),
        )(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation, name=name)(x)
        return x


def upsampleSum(x, conv, filters=128, ratio=0.5, activation=ACTIVATION, name=None):
    """
    Upsample convolution layer and average it with another one with same shape.
    :param x: output from conv layer that should be upsampled
    :param conv: convolution layer to average
    :param filters: number of output filters, defaults to 128
    :param ratio: ratio of the sum
    :param name: name of the scope
    :return: upsampled and merged output
    """
    with tf.name_scope(name) as scope:
        x_up = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="nearest")(x)
        p = (1.0 - ratio) * x_up + ratio * conv2DBatchNorm(
            conv, filters=filters, kernel_size=(1, 1), name="upsum" + name
        )
        return p


def set_regularization(model, kernel_regularizer=None, bias_regularizer=None):
    """
    Adds regularization to the all layers of the base model.
    :param model: base model
    :param kernel_regularizer: tf.keras.regularizers.*, defaults to None
    :param bias_regularizer: tf.keras.regularizers*, defaults to None
    :return: modified model with regularizations
    """
    for layer in model.layers:
        if kernel_regularizer is not None and hasattr(layer, "kernel_regularizer"):
            layer.kernel_regularizer = kernel_regularizer

        if bias_regularizer is not None and hasattr(layer, "bias_regularizer"):
            layer.bias_regularizer = bias_regularizer

    out = tf.keras.models.model_from_json(model.to_json())
    out.set_weights(model.get_weights())
    return out


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    Replace BatchNormalization layers with this new layer.
    This layer has fixed momentum between (0.9,0.99)
    """

    def __init__(self, momentum=BN_MOMENTUM, name=None, **kwargs):
        super(BatchNormalization, self).__init__(momentum=momentum, name=name, **kwargs)

    def call(self, inputs, training=None):
        return super().call(inputs=inputs, training=training)

    def get_config(self):
        config = super(BatchNormalization, self).get_config()
        return config


class GroupNormalization(tfa.layers.GroupNormalization):
    def __init__(self, momentum=0, groups=8, name=None, **kwargs):
        super(GroupNormalization, self).__init__(groups=8, name=name, **kwargs)

    def call(self, inputs):
        return super().call(inputs=inputs)

    def get_config(self):
        config = super(BatchNormalization, self).get_config()
        return config
