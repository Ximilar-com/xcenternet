import math
import numpy as np
import tensorflow as tf


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma_x=1, sigma_y=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    # h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma_x=diameter / 6, sigma_y=diameter / 6)

    y, x = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_truncate_gaussian(heatmap, center, h_radius, w_radius, k=1):
    h, w = 2 * h_radius + 1, 2 * w_radius + 1
    sigma_x = w / 6
    sigma_y = h / 6
    gaussian = gaussian2D((h, w), sigma_x=sigma_x, sigma_y=sigma_y)

    y, x = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, w_radius), min(width - x, w_radius + 1)
    top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[h_radius - top : h_radius + bottom, w_radius - left : w_radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_heatmap(shape, bboxes, labels):
    heat_map = np.zeros(shape, dtype=np.float32)
    for bbox, cls_id in zip(bboxes, labels):
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        if h > 0 and w > 0:
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))
            ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(heat_map[:, :, cls_id], ct_int, radius)

    return heat_map


def draw_heatmaps(shape, bboxes, labels):
    heat_map = np.zeros(shape, dtype=np.float32)
    for b in range(shape[0]):
        for bbox, cls_id in zip(bboxes[b], labels[b]):
            w, h = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_umich_gaussian(heat_map[b, :, :, cls_id], ct_int, radius)

    return heat_map


def radius_ttf(bbox, h, w):
    alpha = 0.54
    h_radiuses_alpha = int(h / 2.0 * alpha)
    w_radiuses_alpha = int(w / 2.0 * alpha)
    return max(1, h_radiuses_alpha), max(1, w_radiuses_alpha)


import sys

np.set_printoptions(threshold=sys.maxsize)


def get_pred_wh(shape):
    h, w = shape
    base_step = 1
    shifts_x = np.arange(0, (w - 1) * base_step + 1, base_step, dtype=np.float32)
    shifts_y = np.arange(0, (h - 1) * base_step + 1, base_step, dtype=np.float32)
    shift_y, shift_x = np.meshgrid(shifts_y, shifts_x)
    base_loc = np.stack((shift_x, shift_y), axis=0)
    return base_loc


def move_points(center, pct, index, field_1, field_2):
    # compute the aspect ratio of height and width
    move = center[field_1] / center[field_2]
    # move it to just one dominant side
    if center["center"][index] < pct["center"][index]:
        center["center"][index] = center["center"][index] - pct[field_1] * (move / 2)
        pct["center"][index] = pct["center"][index] + pct[field_1] / 2
    else:
        center["center"][index] = center["center"][index] + pct[field_1] * (move / 2)
        pct["center"][index] = pct["center"][index] - pct[field_1] / 2


def draw_heatmaps_ttf(shape, bboxes, labels, fix_collisions=False):
    heat_map = np.zeros(shape, dtype=np.float32)
    box_target = np.ones((shape[0], shape[1], shape[2], 4), dtype=np.float32)
    reg_weight = np.zeros((shape[0], shape[1], shape[2], 1), dtype=np.float32)
    box_target_offset = np.zeros((shape[0], shape[1], shape[2], 4), dtype=np.float32)

    meshgrid = get_pred_wh((shape[1], shape[2]))

    # segmentation
    GRID_SIZE = 24
    seg_cat = np.zeros((shape[0], GRID_SIZE, GRID_SIZE, shape[3]), dtype=np.float32)

    # go over batch of images
    for b in range(shape[0]):
        centers = []

        # sort the boxes by the area from max to min
        areas = np.asarray([bbox_areas_log_np(np.asarray(bbox)) for bbox in bboxes[b]])
        indices = np.argsort(-areas)

        bboxes_new = bboxes[b][indices]
        labels_new = labels[b][indices]

        i = 0
        for bbox, cls_id in zip(bboxes_new, labels_new):
            bbox = np.asarray(bbox)
            area = bbox_areas_log_np(bbox)
            w, h = bbox[3] - bbox[1], bbox[2] - bbox[0]

            if w > 0 and h > 0:
                h_radius, w_radius = radius_ttf(bbox, h, w)
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                centers.append({"center": ct, "h_radius": h_radius, "w_radius": w_radius, "area": area, "index": i, "w": w, "h": h, "skip": False})
            else:
                # will be skipped
                centers.append({"h": 0, "w": 0, "index": i, "skip":True})
            i += 1

        if fix_collisions:
            # objects are sorted from biggest area to minimum
            for bbox, cls_id, center in zip(bboxes_new, labels_new, centers):
                if center["h"] > 0 and center["w"] > 0:
                    # heuristic for moving points when there is a collision in centers
                    for pct in centers:
                        # if we compare same object then skip
                        if center["index"] == pct["index"] or pct["skip"]:
                            continue

                        # computes ratios (height, width) between two rectangles
                        h_radius_r, w_radius_r = center["h_radius"] / (pct["h_radius"]), center["w_radius"] / (pct["w_radius"])
                        # sort in which direction we want to object first (height [y] or width [x])
                        fields = ["h_radius", "w_radius"] if h_radius_r > w_radius_r else ["w_radius", "h_radius"]
                        for field in fields:
                            # compute distance between two centers
                            distance = np.linalg.norm(pct["center"] - center["center"])
                            # if distance between two centers is smaller than radius of heatmap of analysis object
                            if distance < center[field]/2:
                                # then move the centers from each other
                                s_field = "w_radius" if field == "h_radius" else "h_radius"
                                index = 0 if field == "h_radius" else 1
                                move_points(center, pct, index, field, s_field)

        # and now compute the heatmaps
        for bbox, cls_id, center in zip(bboxes_new, labels_new, centers):
            bbox = np.asarray(bbox)
            fake_heatmap = np.zeros((shape[1], shape[2]))

            if center["h"] > 0 and center["w"] > 0:
                draw_truncate_gaussian(fake_heatmap, center["center"].astype(np.int32), center["h_radius"], center["w_radius"])
                heat_map[b, :, :, cls_id] = np.clip(np.maximum(heat_map[b, :, :, cls_id], fake_heatmap), 0.0, 1.0)

                # computes indices where is the current heatmap
                box_target_inds = fake_heatmap > 0

                # compute bbox size for current heatmap of bbox
                box_target[b, box_target_inds, :] = bbox[:]

                # this is just for debug/test
                box_target_offset[b, box_target_inds, 0] = meshgrid[1, box_target_inds] - bbox[0]
                box_target_offset[b, box_target_inds, 1] = meshgrid[0, box_target_inds] - bbox[1]
                box_target_offset[b, box_target_inds, 2] = bbox[2] - meshgrid[1, box_target_inds]
                box_target_offset[b, box_target_inds, 3] = bbox[3] - meshgrid[0, box_target_inds]

                # compute weight map for current heatmap of bbox
                local_heatmap = fake_heatmap[box_target_inds]
                ct_div = local_heatmap.sum()
                local_heatmap *= center["area"]
                reg_weight[b, box_target_inds, 0] = local_heatmap / ct_div

        # segmentation categories map for solo
        cat_shape = tf.squeeze(tf.constant([GRID_SIZE, GRID_SIZE]))
        seg_centers = tf.cast(
            tf.constant(
                [[center["center"].astype(np.int32)[0] / shape[1] * GRID_SIZE, center["center"].astype(np.int32)[0] / shape[2] * GRID_SIZE] for center in centers if center["skip"] == False]
            ),
            dtype=tf.int32
        )
        cat = tf.scatter_nd(seg_centers, 
            [cls_id + 1 for center, cls_id in zip(centers, labels_new) if center["skip"] == False],
            cat_shape
        ) - 1                                                     # shape (S, S)
        cat = tf.one_hot(cat, shape[3], dtype=tf.float32)         # shape (S, S, C)
        seg_cat[b] = cat.numpy()

    return heat_map, box_target, reg_weight, box_target_offset, seg_cat


def bbox_areas_log_np(bbox):
    x_min, y_min, x_max, y_max = bbox[1], bbox[0], bbox[3], bbox[2]
    area = (y_max - y_min + 1) * (x_max - x_min + 1)
    return np.log(area)
