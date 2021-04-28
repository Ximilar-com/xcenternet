import numpy as np
import tensorflow as tf
import unittest
from xcenternet.model.evaluation.overlap import compute_overlap

from xcenternet.model.evaluation.mean_average_precision import MAP


class TestMeanAveragePrecision(unittest.TestCase):
    def setUp(self):
        self.map_bboxes = np.array(
            [
                [[20, 10, 80, 60], [10, 40, 40, 90], [0, 0, 100, 100]],
                [[0, 0, 10, 10], [20, 20, 40, 90], [80, 20, 100, 50]],
            ],
            dtype=np.float64,
        )
        self.map_labels = np.array([[0, 0, 1], [0, 0, 0]])
        self.map_predictions = np.array(
            [
                [
                    [10, 40, 40, 90, 0.1, 0],  # overlap 1.00 with bbox #2, low prob
                    [60, 10, 90, 60, 0.5, 0],  # overlap 0.29 with bbox #1
                    [10, 30, 50, 90, 0.7, 0],  # overlap 0.625 with bbox #2
                    [0, 0, 100, 90, 0.7, 1],  # overlap 0.9 with bbox #3
                    [0, 0, 100, 80, 0.7, 1],  # overlap 0.8 with bbox #3
                ],
                [
                    [20, 20, 30, 50, 0.6, 0],  # 0.21 overlap with #2
                    [2, 0, 10, 11, 0.8, 0],  # overlap with #1
                    [0, 2, 14, 10, 0.9, 0],  # overlap with #1
                    [0, 0, 10, 10, 0.7, 1],  # no ground truth for 1
                    [80, 20, 100, 50, 0.1, 1],  # no ground truth for 1
                ],
            ],
            dtype=np.float32,
        )
        self.map_masks = np.array([[1, 1, 1], [1, 1, 1]], dtype=np.float32)

        self.result_1 = {"overall": 3 / 4, "weighted": 2 / 3, "per_class": {0: (0.5, 2), 1: (1.0, 1)}}
        self.result_both = {"overall": 2 / 3, "weighted": 4 / 9, "per_class": {0: (1 / 3, 5), 1: (1.0, 1)}}

    def test_compute_overlap(self):
        boxes1 = np.array([[10, 10, 30, 50], [10, 10, 30, 30]], dtype=np.float64)
        boxes2 = np.array([[10, 10, 30, 50], [10, 10, 40, 40], [100, 70, 110, 90]], dtype=np.float64)

        overlap = compute_overlap(boxes1, boxes2)
        self.assertAlmostEqual(1.0, overlap[0][0])
        self.assertAlmostEqual(6 / 11, overlap[0][1])
        self.assertAlmostEqual(0.0, overlap[0][2])
        self.assertAlmostEqual(0.5, overlap[1][0])
        self.assertAlmostEqual(4 / 9, overlap[1][1])
        self.assertAlmostEqual(0.0, overlap[1][2])

    def test_map_update_one(self):
        mean_average_precision = MAP(2, iou_threshold=0.5, score_threshold=0.3)
        mean_average_precision.update_state(self.map_predictions[0], self.map_bboxes[0], self.map_labels[0])

        result = mean_average_precision.result()
        self._assert_map(result, self.result_1)

    def test_map_update_both(self):
        mean_average_precision = MAP(2, iou_threshold=0.5, score_threshold=0.3)
        mean_average_precision.update_state(self.map_predictions[0], self.map_bboxes[0], self.map_labels[0])
        mean_average_precision.update_state(self.map_predictions[1], self.map_bboxes[1], self.map_labels[1])

        result = mean_average_precision.result()
        self._assert_map(result, self.result_both)

    def test_map_update_batch_one(self):
        mean_average_precision = MAP(2, iou_threshold=0.5, score_threshold=0.3)
        mean_average_precision.update_state_batch(
            tf.constant([self.map_predictions[0]]),
            tf.constant([self.map_bboxes[0]]),
            tf.constant([self.map_labels[0]]),
            tf.constant([self.map_masks[0]]),
        )

        result = mean_average_precision.result()
        self._assert_map(result, self.result_1)

    def test_map_update_batch_both(self):
        mean_average_precision = MAP(2, iou_threshold=0.5, score_threshold=0.3)
        mean_average_precision.update_state_batch(
            tf.constant(self.map_predictions),
            tf.constant(self.map_bboxes),
            tf.constant(self.map_labels),
            tf.constant(self.map_masks),
        )

        result = mean_average_precision.result()
        self._assert_map(result, self.result_both)

    def _assert_map(self, first, second):
        self.assertAlmostEqual(first["overall"], second["overall"])
        self.assertAlmostEqual(first["weighted"], second["weighted"])
        self.assertAlmostEqual(first["per_class"][0][0], second["per_class"][0][0])  # mAP
        self.assertAlmostEqual(first["per_class"][0][1], second["per_class"][0][1])  # num objects
        self.assertAlmostEqual(first["per_class"][1][0], second["per_class"][1][0])  # mAP
        self.assertAlmostEqual(first["per_class"][1][1], second["per_class"][1][1])  # num objects


if __name__ == "__main__":
    unittest.main()
