import tensorflow as tf


class XTensorBoardCallback(tf.keras.callbacks.TensorBoard):
    """
    TensorBoard logging with a learning rate added.
    """

    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({"lr": tf.keras.backend.get_value(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)

    def on_batch_end(self, batch, logs=None):
        logs.update({"lr": tf.keras.backend.get_value(self.model.optimizer.lr)})
        super().on_batch_end(batch, logs)
