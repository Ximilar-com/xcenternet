import tensorflow as tf


def freeze_base(model):
    """
    Our models are build in such way that a backbone model is a layer in the final model.
    This function finds the model and freezes it,

    Do not forget to compile your model after freezing!

    :param model: model which base will be frozen
    :return Whether the base was found and frozen.
    """
    return unfreeze_base(model, 0)


def unfreeze_base(model, fraction=1.0):
    """
    Make the full model trainable. If there is a model inside the the layers, all is unfroze.

    Do not forget to compile your model after freezing!

    :param model: model which base will be (partially) frozen
    :param fraction: how many layers from top should be unfroze
    :return Whether the base was found and frozen.
    """
    base = None
    for id, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.Model):
            base = layer
            break

    if not base:
        return False

    frozed_layers = int(len(base.layers) * (1.0 - fraction))
    for id, layer in enumerate(base.layers):
        layer.trainable = id >= frozed_layers

    return True
