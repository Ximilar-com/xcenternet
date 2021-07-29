import numpy as np
import tensorflow as tf


def visualize_heatmap(heatmaps):
    """
    Draw all heatmaps from a single picture. Heatmaps are placed into a grid, starting on left top corner and
    filling the rows as we were writing from left to right, top to bottom.

    :param heatmaps: Heatmap for a single picture, shape (size, size, labels)
    :return: picture (2D numpy array) with all heatmaps
    """
    labels = heatmaps.shape[-1]
    heatmap_size = heatmaps.shape[0]

    # grid_size
    horizontal = int(np.ceil(np.sqrt(labels)))
    vertical = int(np.ceil(float(labels) / horizontal))

    # resulting image size
    image_shape = ((heatmap_size + 1) * vertical, (heatmap_size + 1) * horizontal)

    result = np.ones(image_shape)
    for id in range(labels):
        y, x = id // horizontal, id % horizontal
        y_start, x_start = y * (heatmap_size + 1), x * (heatmap_size + 1)
        result[y_start : y_start + heatmap_size, x_start : x_start + heatmap_size] = heatmaps[:, :, id]

    return result


def draw_bounding_boxes(image, bboxes, labels, config):
    """
    Draw bounding boxes inside given image. Use colors according to the label and config.
    (To have the colors same across images and epochs.)

    :param image: 3-D Tensor of shape [height, width, channels], range 0 - 255
    :param bboxes:  2-D Tensor (box_number, 4) containing bounding boxes in format [ymin, xmin, ymin, xmax], relative c.
    :param labels: 1-D Tensor (box_number, 1) containing bounding box labels
    :param config: training configuration containing colors for the bounding boxes
    :return: image with bounding boxes
    """
    image = tf.cast(image, dtype=tf.float32)

    # draw bounding boxes on image
    if len(bboxes) > 0:
        colors = np.array([config.debug_class_colors[int(label)] for label in labels.numpy()])
        image = tf.expand_dims(image, 0)
        image = tf.image.draw_bounding_boxes(image, [bboxes], colors)[0]

    return image


def draw_heatmaps(image, heatmaps, config):
    for i in tf.range(tf.shape(heatmaps)[-1]):
        heatmap_image = heatmaps[:, :, i]
        heatmap_image = tf.expand_dims(heatmap_image, -1)
        heatmap_image = tf.image.grayscale_to_rgb(heatmap_image)
        heatmap_image = tf.image.resize(heatmap_image, (tf.shape(image)[0], tf.shape(image)[1]))

        # dim the image behind the heatmap a bit
        image *= 1.0 - tf.clip_by_value(heatmap_image * 1.1, 0.0, 1.0)

        # add the heatmap to the image
        image += heatmap_image * config.debug_class_colors[i] * 0.75

    image = tf.clip_by_value(image, 0.0, 255.0)
    return image


def draw_segmaps(image, segmaps, config):
    for i in tf.range(tf.shape(segmaps)[-1]):
        heatmap_image = segmaps[:, :, i]
        heatmap_image = tf.expand_dims(heatmap_image, -1)
        heatmap_image = tf.image.grayscale_to_rgb(heatmap_image)
        heatmap_image = tf.image.resize(heatmap_image, (tf.shape(image)[0], tf.shape(image)[1]))

        # dim the image behind the heatmap a bit
        image *= 1.0 - tf.clip_by_value(heatmap_image * 1.1, 0.0, 1.0)

        # add the heatmap to the image
        image += heatmap_image * [255.0, 255.0, 255.0] * 0.75

    image = tf.clip_by_value(image, 0.0, 255.0)
    return image
