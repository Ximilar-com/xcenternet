import tensorflow as tf

from xcenternet.model.config import XModelType


def nms(heat, kernel=3):
    hmax = tf.keras.layers.MaxPooling2D(kernel, 1, padding="same")(heat)
    keep = tf.cast(tf.equal(heat, hmax), tf.float32)
    return heat * keep


def topk(hm, k=100):
    batch, height, width, cat = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
    scores = tf.reshape(hm, (batch, -1))
    topk_scores, topk_inds = tf.nn.top_k(scores, k=k)

    topk_clses = topk_inds % cat
    topk_xs = tf.cast(topk_inds // cat % width, tf.float32)
    topk_ys = tf.cast(topk_inds // cat // width, tf.float32)
    topk_inds = tf.cast(topk_ys * tf.cast(width, tf.float32) + topk_xs, tf.int32)
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def decode(model_type: XModelType, heat, wh, reg=None, k=100, relative=False):
    batch, height, width, cat = tf.shape(heat)[0], tf.shape(heat)[1], tf.shape(heat)[2], tf.shape(heat)[3]
    heat = nms(heat)
    scores, inds, clses, ys, xs = topk(heat, k=k)

    ys = tf.expand_dims(ys, axis=-1)
    xs = tf.expand_dims(xs, axis=-1)

    if model_type == XModelType.CENTERNET and reg is not None:
        reg = tf.reshape(reg, (batch, -1, tf.shape(reg)[-1]))
        reg = tf.gather(reg, inds, axis=1, batch_dims=-1)
        ys, xs = ys + reg[..., 0:1], xs + reg[..., 1:2]

    wh = tf.reshape(wh, (batch, -1, tf.shape(wh)[-1]))
    wh = tf.gather(wh, inds, axis=1, batch_dims=-1)

    clses = tf.cast(tf.expand_dims(clses, axis=-1), tf.float32)
    scores = tf.expand_dims(scores, axis=-1)

    # Especially in the early phases of the training, the values can be negative, Be careful.
    # (The ways of solving this does not matter much ff the network is learning.)
    if model_type == XModelType.CENTERNET:
        wh = tf.math.abs(wh)
        ymin = ys - wh[..., 0:1] / 2
        xmin = xs - wh[..., 1:2] / 2
        ymax = ys + wh[..., 0:1] / 2
        xmax = xs + wh[..., 1:2] / 2
    elif model_type == XModelType.TTFNET:
        ymin = ys - wh[..., 0:1]
        xmin = xs - wh[..., 1:2]
        ymax = ys + wh[..., 2:3]
        xmax = xs + wh[..., 3:4]
    else:
        raise ValueError(f"Unsupported model type {model_type}!")

    bboxes = tf.concat([ymin, xmin, ymax, xmax], axis=-1)

    if relative:
        bboxes /= tf.cast(tf.stack([height, width, height, width]), dtype=tf.float32)

    detections = tf.concat([bboxes, scores, clses], axis=-1)
    return detections


def filter_detections(detections, score_threshold=0.3, nms_iou_threshold=0.5, max_size=None, class_nms=False):
    """
    Remove detections with a low probability, run Non-maximum Suppression. If set, apply max size to all boxes.

    :param detections: Tensor with detections in form [ymin, xmin, ymax, xmax, label, probability].
    :param score_threshold: Minimal probability of a detection to be included in the result.
    :param nms_iou_threshold: Threshold for deciding whether boxes overlap too much with respect to IOU.
    :param max_size: Max size to strip the given bounding boxes, default: None = no stripping.
    :param class_nms: If True use nms per classes (default False)
    :return: Filtered bounding boxes.
    """
    mask = detections[:, 4] >= score_threshold
    result = tf.boolean_mask(detections, mask)
    bboxes, labels, scores = result[:, 0:4], result[:, 5], result[:, 4]

    if max_size is not None:
        bboxes = tf.clip_by_value(tf.cast(bboxes, tf.float32), 0.0, max_size)

    if tf.shape(bboxes)[0] == 0:
        return tf.constant([], shape=(0, 6))

    if class_nms:
        unique_labels, _ = tf.unique(labels)
        detection_results = []
        for label in unique_labels:
            y = tf.where(result[:, 5] == label)

            c_bboxes = tf.gather_nd(bboxes, y)
            c_labels = tf.gather_nd(labels, y)
            c_scores = tf.gather_nd(scores, y)

            c_bboxes = tf.clip_by_value(tf.cast(c_bboxes, tf.float32), 0.0, max_size)

            indices = tf.image.non_max_suppression(c_bboxes, c_scores, tf.shape(c_bboxes)[0], iou_threshold=nms_iou_threshold)
            selected_boxes = tf.gather(c_bboxes, indices)
            selected_labels = tf.gather(c_labels, indices)
            selected_scores = tf.gather(c_scores, indices)
            detection_results.append(tf.concat(
                [selected_boxes, tf.expand_dims(selected_scores, axis=1), tf.expand_dims(selected_labels, axis=1)], axis=1
            ))
            # todo: fix sorting
        return tf.concat(detection_results, axis=0)

    max_objects = tf.shape(result)[0]
    selected_indices = tf.image.non_max_suppression(bboxes, scores, max_objects, iou_threshold=nms_iou_threshold)
    selected_boxes = tf.gather(bboxes, selected_indices)
    selected_labels = tf.gather(labels, selected_indices)
    selected_scores = tf.gather(scores, selected_indices)

    detections = tf.concat(
        [selected_boxes, tf.expand_dims(selected_scores, axis=1), tf.expand_dims(selected_labels, axis=1)], axis=1
    )
    return detections
