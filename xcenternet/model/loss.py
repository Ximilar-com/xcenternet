import tensorflow as tf
from xcenternet.model.constants import SOLO_GRID_SIZE


def heatmap_focal_loss(outputs, training_data, predictions):
    return focal_loss(outputs["heatmap"], predictions[0])


def size_l1_loss(outputs, training_data, predictions):
    heatmap_size = outputs["size"]
    indices = training_data["indices"]
    mask = training_data["mask"]

    return 0.1 * reg_l1_loss(heatmap_size, predictions[1], indices, mask)


def offset_l1_loss(outputs, training_data, predictions):
    local_offset = outputs["offset"]
    indices = training_data["indices"]
    mask = training_data["mask"]

    return reg_l1_loss(local_offset, predictions[2], indices, mask)


def giou_loss(outputs, training_data, predictions):
    box_target = training_data["box_target"]
    reg_weight = training_data["reg_weight"]

    # the 5.0 as weight of giou loss was taken as from original TTFNet pape
    # for negative images we sometimes get nan values
    value = 5.0 * compute_giou_loss(box_target, reg_weight, predictions[1])
    value_not_nan = tf.dtypes.cast(tf.math.logical_not(tf.math.is_nan(value)), dtype=tf.float32)
    return tf.math.multiply_no_nan(value, value_not_nan)


@tf.function
def outerprodflatten(x, y, channel_dims):
    """
    According to StackOverflow:
    https://stackoverflow.com/questions/68361071/multiply-outputs-from-two-conv2d-layers-in-tensorflow-2
    """
    return tf.repeat(x,channel_dims,-1)*tf.tile(y,[1,1,1,channel_dims])


def solo_loss_cate(outputs, training_data, predictions):
    """
    SOLO Decoupled head loss for category branch in TF2.
    """
    seg_cate = outputs["seg_cate"]
    l_cate = focal_loss_segmentation(seg_cate, predictions[2])
    return l_cate


def solo_loss_mask(outputs, training_data, predictions):
    """
    SOLO Decoupled head loss for two mask branches in TF2.
    """
    seg_mask = outputs["seg_mask"]

    # from decoupled head 2x(b, size, size, 24) to (b, size, size, 24*24)
    mask_preds = outerprodflatten(predictions[3], predictions[4], SOLO_GRID_SIZE)

    # now segmentation mask loss
    l_mask = solo_mask_loss(seg_mask, mask_preds)
    return tf.reduce_sum(l_mask)


@tf.function
def compute_giou_loss(box_target, wh_weight, pred_wh, mode="diou", reduce="sum"):
    """
    Computes giou loss and in future also diou or ciou loss for ttfnet.
    :param box_target: ground truth bounding boxes
    :param wh_weight: weight of heatmap
    :param pred_wh: prediction of 4 values (offsets to left upper and right bottom corner)
    :param mode: giou or diou or ciou, defaults to "diou"
    :param reduce: sum over batch or mean the batch loss, defaults to "sum""
    :return: Computes giou loss.
    """
    base_step = 1
    b = tf.shape(wh_weight)[0]
    h = tf.shape(wh_weight)[1]
    w = tf.shape(wh_weight)[2]
    mask = tf.reshape(wh_weight, (b, h, w))
    avg_factor = tf.reduce_sum(mask)
    pos_mask = mask > 0.0
    weight = tf.cast(mask[pos_mask], tf.float32)

    shifts_x = tf.range(0, (w - 1) * base_step + 1, base_step, dtype=tf.float32)
    shifts_y = tf.range(0, (h - 1) * base_step + 1, base_step, dtype=tf.float32)

    shift_y, shift_x = tf.meshgrid(shifts_y, shifts_x, indexing="ij")
    base_loc = tf.stack((shift_y, shift_x), axis=0)
    base_loc = tf.cast(base_loc, tf.float32)

    y1 = base_loc[0] - pred_wh[:, :, :, 0]
    x1 = base_loc[1] - pred_wh[:, :, :, 1]
    y2 = base_loc[0] + pred_wh[:, :, :, 2]
    x2 = base_loc[1] + pred_wh[:, :, :, 3]

    pred_wh = tf.stack((y1, x1, y2, x2), axis=3)

    bboxes2 = pred_wh[pos_mask]
    bboxes1 = box_target[pos_mask]

    bboxes_num_per_sample = tf.math.reduce_sum(tf.cast(pos_mask, dtype=tf.int32), axis=[1, 2])

    losses = _giou_loss(bboxes1, bboxes2, mode=mode)

    if reduce == "mean":
        return tf.math.reduce_mean(losses * weight) / avg_factor
    elif reduce == "sum":
        return tf.math.reduce_sum(losses * weight) / avg_factor
    return (losses * weight) / avg_factor, bboxes_num_per_sample


@tf.function
def focal_loss(hm_true, hm_pred):
    """
    Computes focal loss for heatmap.

    This function was taken from:
        https://github.com/MioChiu/TF_CenterNet/blob/master/loss.py

    :param hm_true: gt heatmap
    :param hm_pred: predicted heatmap
    :return: loss value
    """
    pos_mask = tf.cast(tf.equal(hm_true, 1.0), dtype=tf.float32)
    neg_mask = tf.cast(tf.less(hm_true, 1.0), dtype=tf.float32)
    neg_weights = tf.pow(1.0 - hm_true, 4)

    pos_loss = -tf.math.log(tf.clip_by_value(hm_pred, 1e-5, 1.0 - 1e-5)) * tf.math.pow(1.0 - hm_pred, 2.0) * pos_mask
    neg_loss = (
        -tf.math.log(tf.clip_by_value(1.0 - hm_pred, 1e-5, 1.0 - 1e-5))
        * tf.math.pow(hm_pred, 2.0)
        * neg_weights
        * neg_mask
    )

    num_pos = tf.reduce_sum(pos_mask)
    pos_loss = tf.reduce_sum(pos_loss)
    neg_loss = tf.reduce_sum(neg_loss)

    loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)
    return loss


@tf.function
def reg_l1_loss(y_true, y_pred, indices, mask):
    """
    This function was taken from:
        https://github.com/MioChiu/TF_CenterNet/blob/master/loss.py

    :param y_true: (batch, max_objects, 2)
    :param y_pred: (batch, heatmap_height, heatmap_width, max_objects)
    :param indices: (batch, max_objects)
    :param mask: (batch, max_objects)
    :return: l1 loss (single float value) for given predictions and ground truth
    """
    batch_dim = tf.shape(y_pred)[0]
    channel_dim = tf.shape(y_pred)[-1]
    y_pred = tf.reshape(y_pred, (batch_dim, -1, channel_dim))
    indices = tf.cast(indices, tf.int32)
    y_pred = tf.gather(y_pred, indices, batch_dims=1)
    mask = tf.tile(tf.expand_dims(mask, axis=-1), (1, 1, 2))
    total_loss = tf.reduce_sum(tf.abs(y_true * mask - y_pred * mask))
    loss = total_loss / (tf.reduce_sum(mask) + 1e-5)
    return loss


def _giou_loss(b1, b2, mode):
    """
    Args:
        b1: bounding box. The coordinates of the each bounding box in boxes are
            encoded as [y_min, x_min, y_max, x_max].
        b2: the other bounding box. The coordinates of the each bounding box
            in boxes are encoded as [y_min, x_min, y_max, x_max].
        mode: one of ['iou', 'ciou', 'diou', 'giou'], decided to calculate IoU or CIoU or DIoU or GIoU.
    Returns:
        IoU loss float `Tensor`.
    """

    zero = 0.0
    b1_ymin, b1_xmin, b1_ymax, b1_xmax = tf.unstack(b1, 4, axis=-1)
    b2_ymin, b2_xmin, b2_ymax, b2_xmax = tf.unstack(b2, 4, axis=-1)
    b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
    b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
    b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
    b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
    b1_area = b1_width * b1_height
    b2_area = b2_width * b2_height

    intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
    intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
    intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
    intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
    intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
    intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
    intersect_area = intersect_width * intersect_height

    union_area = b1_area + b2_area - intersect_area
    iou = tf.math.divide_no_nan(intersect_area, union_area)
    if mode == "iou":
        return 1.0 - iou

    elif mode in ["diou"]:
        enclose_ymin = tf.minimum(b1_ymin, b2_ymin)
        enclose_xmin = tf.minimum(b1_xmin, b2_xmin)
        enclose_ymax = tf.maximum(b1_ymax, b2_ymax)
        enclose_xmax = tf.maximum(b1_xmax, b2_xmax)

        b1_center = tf.stack([(b1_ymin + b1_ymax) / 2, (b1_xmin + b1_xmax) / 2])
        b2_center = tf.stack([(b2_ymin + b2_ymax) / 2, (b2_xmin + b2_xmax) / 2])
        euclidean = tf.linalg.norm(b2_center - b1_center)
        diag_length = tf.linalg.norm([enclose_ymax - enclose_ymin, enclose_xmax - enclose_xmin])
        diou = iou - (euclidean ** 2) / (diag_length ** 2)
        return 1.0 - diou
    elif mode == "giou":
        enclose_ymin = tf.minimum(b1_ymin, b2_ymin)
        enclose_xmin = tf.minimum(b1_xmin, b2_xmin)
        enclose_ymax = tf.maximum(b1_ymax, b2_ymax)
        enclose_xmax = tf.maximum(b1_xmax, b2_xmax)
        enclose_width = tf.maximum(zero, enclose_xmax - enclose_xmin)
        enclose_height = tf.maximum(zero, enclose_ymax - enclose_ymin)
        enclose_area = enclose_width * enclose_height
        giou = iou - tf.math.divide_no_nan((enclose_area - union_area), enclose_area)
        return 1.0 - giou
    else:
        raise ValueError("Value of mode should be one of ['iou','giou','diou']")


def dice_loss(y_true, y_pred, keepdims=True):
    pq = tf.math.reduce_sum(tf.math.multiply(y_pred, y_true), axis=[1,2], keepdims=keepdims)
    p2 = tf.math.reduce_sum(tf.math.multiply(y_pred, y_pred), axis=[1,2], keepdims=keepdims)
    q2 = tf.math.reduce_sum(tf.math.multiply(y_true, y_true), axis=[1,2], keepdims=keepdims)
    return 1 - 2 * pq / (p2 + q2)   # shape (B, 1, 1, S^2) if keepdims else (B, S^2)


def focal_loss_segmentation(y_true, y_pred):
    epsilon, alpha, gamma = 1e-7, 0.25, 2.0
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    loss = -alpha * tf.math.pow(1 - y_pred, gamma) * y_true * tf.math.log(y_pred)
    return tf.math.reduce_sum(loss)


def solo_mask_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    d_mask = dice_loss(y_true, y_pred)                                     # shape (B, x, x, S^2)
    d_mask = tf.math.reduce_mean(d_mask, axis=[1,2])                            # shape (B, S^2)
    indicator = tf.cast(tf.math.reduce_sum(y_true, axis=[1,2]) > 0, tf.float32)
    n_pos = tf.math.reduce_sum(indicator, axis=1)
    n_pos = tf.math.maximum(n_pos, tf.ones_like(n_pos, dtype=tf.float32)) # n_pos.shape, dtype=tf.float32))      # shape (B,), prevent divided by 0
    loss = tf.math.reduce_sum(indicator * d_mask, axis=1) / n_pos               # shape (B,)
    return 3.0 * loss
