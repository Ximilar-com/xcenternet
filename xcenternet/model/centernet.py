import tensorflow as tf

from xcenternet.model.config import XModelType
from xcenternet.model.decoder import decode
from xcenternet.model.loss import offset_l1_loss, size_l1_loss, heatmap_focal_loss, giou_loss, solo_loss


class XCenternetModel(tf.keras.Model):
    def __init__(self, *args, segmentation=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.segmentation = segmentation
        self.custom_losses = None

    def compile(self, optimizer="adam", loss=None, metrics=None, **kwargs):
        super().compile(optimizer=optimizer, metrics=metrics)

        if loss is not None:
            self.custom_losses = XCustomLossContainer(**loss if loss else [])

    def train_step(self, data):
        with tf.GradientTape() as tape:
            y_pred = self(data[0]["input"], training=True)
            reg_loss, loss = self.custom_losses.update_state(
                data[1], data[2], y_pred, regularization_losses=self.losses
            )

        gradients = tape.gradient(loss, self.trainable_variables)
        gradients = [(tf.clip_by_norm(grad, 1.0)) for grad in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        losses = {m.name: m.result() for m in self.metrics}
        losses["regularization"] = reg_loss
        return losses

    def test_step(self, data):
        y_pred = self(data[0]["input"], training=False)
        self.custom_losses.update_state(data[1], data[2], y_pred, regularization_losses=self.losses)

        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        metrics = super().metrics
        if self.custom_losses:
            metrics.append(self.custom_losses.loss_sum)
            metrics.extend(self.custom_losses.means)
        return metrics

    def decode(self, predictions, relative, k=100):
        return decode(XModelType.CENTERNET, predictions[0], predictions[1], reg=predictions[2], k=k, relative=relative)

    def get_loss_funcs(self):
        return {"loss_heatmap": heatmap_focal_loss, "loss_size": size_l1_loss, "loss_offset": offset_l1_loss}


class XTTFModel(XCenternetModel):
    def __init__(self, *args, segmentation=False, **kwargs):
        super().__init__(*args, **kwargs)

    def decode(self, predictions, relative, k=100):
        return decode(XModelType.TTFNET, predictions[0], predictions[1], k=k, relative=relative)

    def get_loss_funcs(self):
        # return {"loss_heatmap": heatmap_focal_loss, "loss_giou": giou_loss}
        return {"loss_heatmap": heatmap_focal_loss, "loss_giou": giou_loss, "solo_loss": solo_loss}


# class XTTFSOLOModel(XCenternetModel):
#     def __init__(self, *args, segmentation=False, **kwargs):
#         super().__init__(*args, **kwargs)

#     def decode(self, predictions, relative, k=100):
#         return decode(XModelType.TTFNET, predictions[0], predictions[1], predictions[2], k=k, relative=relative)

#     def get_loss_funcs(self):
#         return {"loss_heatmap": heatmap_focal_loss, "loss_giou": giou_loss, "solo_loss": solo_loss}


class XCustomLossContainer(object):
    def __init__(self, **losses):
        self.losses = list(losses.values())
        self.means = [tf.keras.metrics.Mean(name) for name in losses.keys()]
        self.loss_sum = tf.keras.metrics.Mean("loss")

    def update_state(self, outputs, training_data, predictions, regularization_losses=None):
        result = []

        for loss, mean in zip(self.losses, self.means):
            res = loss(outputs, training_data, predictions)
            tf.print(loss, res)
            result.append(res)
            mean.update_state(res)

        loss = tf.add_n(result)

        if regularization_losses:
            reg_loss = tf.add_n(regularization_losses)
            loss += reg_loss
            self.loss_sum.update_state(loss)
            return reg_loss, loss
        else:
            self.loss_sum.update_state(loss)
            return 0.0, loss
