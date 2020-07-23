import tensorflow as tf

from xcenternet.tensorboard.visualization import draw_bounding_boxes, visualize_heatmap


class ResultImageLogCallback(tf.keras.callbacks.Callback):
    def __init__(self, data, config, pred_model, freq=1, log_dir=None):
        super().__init__()
        self.data = data
        self.config = config
        self.pred_model = pred_model
        self.freq = freq
        self.file_writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        # i need to add this if statement otherwise interrupting training does not work
        if not self.model.stop_training and (epoch % self.freq == 0):
            self.log_image_result(epoch)

    def log_image_result(self, epoch):
        images = []
        heatmaps = []

        # take the first batch
        iterator = iter(self.data)
        inputs, _, _ = next(iterator)
        images_predict = inputs["input"]

        # make predictions and decode them
        prediction = self.pred_model.predict(images_predict)
        decoded = self.pred_model.decode(prediction, relative=True, k=self.config.max_objects)

        for i in range(len(decoded)):
            image = images_predict[i]

            dec = decoded[i, :, :]
            dec_mask = dec[:, 4] >= 0.3
            result = tf.boolean_mask(dec, dec_mask)
            bboxes = result[:, 0:4]
            labels = result[:, 5]

            image = draw_bounding_boxes(image, bboxes, labels, self.config)
            image /= 255.0
            images.append(image)

            heatmap_image = visualize_heatmap(prediction[0][i])
            heatmap_image = tf.expand_dims(heatmap_image, -1)
            heatmaps.append(heatmap_image)

        with self.file_writer.as_default():
            tf.summary.image(f"results from centernet", images, max_outputs=len(decoded), step=epoch)
            tf.summary.image(f"heatmaps from centernet", heatmaps, max_outputs=len(decoded), step=epoch)
