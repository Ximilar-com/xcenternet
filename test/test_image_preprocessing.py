import tensorflow as tf
import unittest

from xcenternet.model.preprocessing.batch_preprocessing import BatchPreprocessing


class TestImagePreprocessing(unittest.TestCase):
    def setUp(self):
        self.bboxes = tf.constant([[[2.0, 4.0, 30.0, 16.0], [12.0, 20.0, 32.0, 35.0], [0.0, 0.0, 30.0, 4.0]]])

    def test_decompose(self):
        decomposed = BatchPreprocessing.decompose_bounding_boxes(self.bboxes, 60, 4)
        sizes, offsets, indices = decomposed

        # check the first bounding box
        tf.debugging.assert_equal(tf.constant([7.0, 3.0]), sizes[0][0])
        tf.debugging.assert_equal(tf.constant([0.0, 0.5]), offsets[0][0])
        tf.debugging.assert_equal(tf.constant(62), indices[0][0])


if __name__ == "__main__":
    unittest.main()
