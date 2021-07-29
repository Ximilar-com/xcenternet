import tensorflow as tf

from xcenternet.tensorboard.visualization import draw_bounding_boxes, draw_heatmaps, draw_segmaps


class ImageLog(tf.keras.callbacks.Callback):
    def __init__(self, data, config, log_dir="logs", segmentation=False):
        super().__init__()
        self.data = data
        self.config = config
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.segmentation = segmentation

    def on_epoch_begin(self, epoch, logs=None):
        iterator = iter(self.data)
        inputs, outputs, training_data = next(iterator)

        images = inputs["input"]
        heatmaps, bounding_box_sizes, segmasks = outputs["heatmap"], outputs["size"], outputs["seg_mask"]
        local_offsets, indices, labels = outputs["offset"], training_data["indices"], training_data["labels"]

        # calculate bounding boxes
        width = tf.cast(tf.shape(images)[2] // self.config.downsample, dtype=tf.float32)
        center = tf.tile(tf.expand_dims(tf.cast(indices, dtype=tf.float32), -1), [1, 1, 2])
        center = tf.map_fn(lambda c: tf.map_fn(lambda i: tf.stack([i[0] // width, i[1] % width]), c), center)
        center = (center + local_offsets) * self.config.downsample
        bboxes = tf.tile(center, [1, 1, 2])
        sizes = (
            (tf.tile(bounding_box_sizes, [1, 1, 2]) / 2) * tf.constant([-1.0, -1.0, 1.0, 1.0]) * self.config.downsample
        )
        bboxes = (bboxes + sizes) / tf.cast(
            (tf.stack([tf.shape(images)[1], tf.shape(images)[2], tf.shape(images)[1], tf.shape(images)[2]])),
            dtype=tf.float32,
        )

        images = tf.cast(images, dtype=tf.float32)
        images_res = [[]] * images.shape[0]
        for i in range(len(images)):
            images_res[i] = draw_heatmaps(images[i], heatmaps[i], self.config)
            images_res[i] = draw_bounding_boxes(images_res[i], bboxes[i], labels[i], self.config)
            images_res[i] /= 255.0

        with self.file_writer.as_default():
            tf.summary.image(f"training examples epoch", images_res, max_outputs=len(images_res), step=epoch)

        images_res = [[]] * images.shape[0]

        for i in range(len(images)):
            images_res[i] = draw_segmaps(images[i], segmasks[i], self.config)
            images_res[i] /= 255.0

        with self.file_writer.as_default():
            tf.summary.image(f"seg maps epoch", images_res, max_outputs=len(images_res), step=epoch)

