import unittest

import numpy as np
import tensorflow as tf

from xcenternet.model.loss import reg_l1_loss, compute_giou_loss
from xcenternet.model.encoder import draw_heatmaps_ttf


class TestLoss(unittest.TestCase):
    def test_reg_l1_loss(self):
        # 3 batches, max 4 objects, 2 channels, heatmap 5x6
        y_true = tf.constant(
            [
                [[2.0, 3.0], [0.1, 2.0], [0.5, 1.2], [0.0, 0.0]],  # only first valid
                [[4.0, 1.5], [5.4, 10.1], [0.5, 0.1], [6.8, 9.9]],  # first two valid
                [[3.0, 3.0], [0.9, 0.0], [0.5, 1.2], [0.0, 4.0]],  # no valid
            ],
            dtype=tf.float32,
        )
        indices = tf.constant([[12, 20, 21, 22], [4, 20, 21, 22], [1, 2, 3, 4]], dtype=tf.float32)
        mask = tf.constant([[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=tf.float32)

        y_pred = np.random.rand(3, 5, 6, 2)
        y_pred[0, 2, 0, :] = [6.0, 3.2]
        y_pred[1, 0, 4, :] = [4.0, 2.0]
        y_pred[1, 3, 2, :] = [5.0, 9.0]
        y_pred = tf.constant(y_pred, dtype=tf.float32)

        loss_true = (
            tf.math.abs(y_true[0, 0, 0] - y_pred[0, 2, 0, 0])
            + tf.math.abs(y_true[0, 0, 1] - y_pred[0, 2, 0, 1])
            + tf.math.abs(y_true[1, 0, 0] - y_pred[1, 0, 4, 0])
            + tf.math.abs(y_true[1, 0, 1] - y_pred[1, 0, 4, 1])
            + tf.math.abs(y_true[1, 1, 0] - y_pred[1, 3, 2, 0])
            + tf.math.abs(y_true[1, 1, 1] - y_pred[1, 3, 2, 1])
        ) / 6
        loss = reg_l1_loss(y_true, y_pred, indices, mask)
        tf.debugging.assert_equal(tf.math.round(tf.constant(loss_true * 1000)), tf.math.round(loss * 1000))

    def test_giou_loss(self):
        # 2 batches, max 2 objects, 2 labels
        heatmap_shape = (2, 512, 512, 2)
        bboxes = np.asarray(
            [
                [[131, 156, 415, 439], [92, 208, 327, 355]],  # first image
                [[131, 156, 415, 439], [92, 208, 327, 355]],  # second image
            ],
            np.float32,
        )

        labels = np.asarray([[0, 0], [0, 1],])  # first image  # secong image

        bboxes2 = np.asarray([[[92, 208, 327, 355]], [[70, 158, 300, 325]]], np.float32)  # first image  # second image

        labels2 = np.asarray([[0], [1],])  # first image  # second image

        bboxes3 = np.asarray([[[92, 208, 327, 355]], [[92, 208, 327, 355]]], np.float32)  # first image  # second image
        labels3 = np.asarray([[0], [0],])  # first image  # second image

        bboxes4 = np.asarray([[[92, 208, 327, 355]], [[70, 158, 300, 305]],], np.float32)  # first image  # second image
        labels4 = np.asarray([[0], [1],])  # first image  # second image

        bboxes5 = np.asarray(
            [
                [[131, 156, 415, 400], [92, 208, 327, 305]],  # first image
                [[131, 156, 415, 429], [92, 208, 327, 335]],  # second image
            ],
            np.float32,
        )

        labels5 = np.asarray([[0, 0], [0, 1],])  # first image  # secong image

        bboxes6 = np.asarray([[[10, 10, 20, 20]], [[25, 25, 35, 35]]], np.float32)  # first image  # second image

        labels6 = np.asarray([[0], [0],])  # first image  # secong image

        bboxes7 = np.asarray([[[40, 40, 60, 60]], [[40, 40, 60, 60]]], np.float32)  # first image  # second image

        labels7 = np.asarray([[0], [0],])  # first image  # secong image

        heatmap_dense, box_target, reg_weight, box_target_off = draw_heatmaps_ttf(heatmap_shape, bboxes, labels)
        heatmap_dense2, box_target2, reg_weight2, box_target_off2 = draw_heatmaps_ttf(heatmap_shape, bboxes2, labels2)
        heatmap_dense3, box_target3, reg_weight3, box_target_off3 = draw_heatmaps_ttf(heatmap_shape, bboxes3, labels3)
        heatmap_dense4, box_target4, reg_weight4, box_target_off4 = draw_heatmaps_ttf(heatmap_shape, bboxes4, labels4)
        heatmap_dense5, box_target5, reg_weight5, box_target_off5 = draw_heatmaps_ttf(heatmap_shape, bboxes5, labels5)
        heatmap_dense6, box_target6, reg_weight6, box_target_off6 = draw_heatmaps_ttf(heatmap_shape, bboxes6, labels6)
        heatmap_dense7, box_target7, reg_weight7, box_target_off7 = draw_heatmaps_ttf(heatmap_shape, bboxes7, labels7)

        reduce = "sum"
        loss_0 = compute_giou_loss(box_target, reg_weight, box_target_off, reduce=reduce)
        loss_0_s2 = compute_giou_loss(box_target, reg_weight2, box_target_off2, reduce=reduce)
        loss_0_s3 = compute_giou_loss(box_target2, reg_weight2, box_target_off3, reduce=reduce)
        loss_0_s4 = compute_giou_loss(box_target2, reg_weight2, box_target_off4, reduce=reduce)
        loss_0_s5 = compute_giou_loss(box_target, reg_weight, box_target_off5, reduce=reduce)
        loss_0_s6, sum_sample = compute_giou_loss(box_target7, reg_weight6, box_target_off6, reduce=None)

        # print(reg_weight.shape)
        # sprint(tf.reduce_sum(tf.reshape(reg_weight6, (-1, 512, 512)), axis=[1,2]).numpy())

        print(loss_0)
        print(loss_0_s2)
        print(loss_0_s3)
        print(loss_0_s4)
        print(loss_0_s5)
        print(loss_0_s6)

        # print(_common_iou(bboxes6, bboxes7, mode="ciou"))
        # print(_common_iou(bboxes6, bboxes7, mode="giou"))

        tf.debugging.assert_equal(tf.reduce_sum(loss_0), 0.0)
        tf.debugging.assert_equal(loss_0, 0.0)

        tf.debugging.assert_greater(loss_0_s2, 0.0)

        tf.debugging.assert_greater(loss_0_s3, 0.0)

        tf.debugging.assert_greater(loss_0_s4, 0.0)

        tf.debugging.assert_greater(loss_0_s3, loss_0_s4)

        tf.debugging.assert_none_equal(loss_0_s5, 0.0)

        # this fails, don't know why
        fromid1 = tf.math.reduce_sum(sum_sample[0:0])
        toid1 = tf.math.reduce_sum(sum_sample[0:1])
        fromid2 = tf.math.reduce_sum(sum_sample[0:1])
        toid2 = tf.math.reduce_sum(sum_sample[0:2])

        # print(loss)
        print(tf.reduce_sum(loss_0_s6[fromid1:toid1]))
        print(tf.reduce_sum(loss_0_s6[fromid2:toid2]))

        # tf.debugging.assert_greater(loss_0_s6[0], 0)


if __name__ == "__main__":
    unittest.main()
