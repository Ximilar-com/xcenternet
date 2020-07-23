import tensorflow as tf

from xcenternet.model.evaluation.mean_average_precision import MAP


class MAPValidationCallback(tf.keras.callbacks.Callback):
    """
    Callback that allows us to calculate Mean Average Precision and write it to TensorBoard.

    Tp be used on validation dataset and it is performed only when validation is performed.
    We recommend not doing so after each epoch since this step takes some time. (Use validation_freq of fit.)

    Validation dataset need to be provided, since callback cannot obtain it any other way.
    """

    def __init__(
        self,
        log_dir: str,
        val_data: tf.data.Dataset,
        model: tf.keras.Model,
        classes: int,
        max_objects: int,
        iou_threshold: float,
        score_threshold: float,
    ):
        super().__init__()
        self.writer = tf.summary.create_file_writer(log_dir)
        self.val_data = val_data
        self.model = model
        self.max_objects = max_objects
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.mean_average_precision = MAP(classes, iou_threshold=iou_threshold, score_threshold=score_threshold)

    def on_epoch_end(self, epoch, logs=None):
        # If we did not perform validation for this epoch, just skip map calculation
        if not any(map(lambda key: key.startswith("val_"), logs.keys())):
            super().on_batch_end(epoch, logs)
            return

        # calculate mAP and add it to the logs
        self.mean_average_precision.reset_states()
        for inputs, outputs, training_data in self.val_data:
            images = inputs["input"]
            mask, bboxes, labels = training_data["mask"], training_data["bboxes"], training_data["labels"]

            batch_predict = self.model.predict_on_batch(images)
            decoded = self.model.decode(batch_predict, relative=False, k=self.max_objects)
            self.mean_average_precision.update_state_batch(decoded, bboxes, labels, mask)
        result = self.mean_average_precision.result()

        with self.writer.as_default():
            tf.summary.scalar("val_map_overall", result["overall"], epoch)
            tf.summary.scalar("val_map_weighted", result["weighted"], epoch)
