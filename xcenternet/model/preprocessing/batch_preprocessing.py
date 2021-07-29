import tensorflow as tf
from typing import List

from tf_image.core.bboxes.resize import resize
from tf_image.core.resize import random_resize
from xcenternet.model.config import ModelConfig, XModelType
from xcenternet.model.encoder import draw_heatmaps, draw_heatmaps_ttf
from xcenternet.model.preprocessing.augmentations import Augmentation
from xcenternet.model.constants import SOLO_GRID_SIZE


class BatchPreprocessing(object):
    def __init__(
        self,
        model_config: ModelConfig,
        train: bool,
        resize_before_augmenting: bool = True,
        augmentations: List[Augmentation] = None,
        segmentation: bool = False,
    ):
        self.model_config = model_config
        self.train = train
        self.resize_before_augmenting = resize_before_augmenting
        self.augmentations = augmentations
        self.segmentation = segmentation

    @tf.function
    def prepare_for_batch(self, image, labels, bboxes, segmentations, image_id=-1):
        """
        All inputs have different dimensions, we need to update them in order to fit the batch,

        Image: Depending on the config, we rescale image to the batch image size (and it will stay the same)
        or maximum batch image size, which is then transformed to randomly selected size in preprocess_batch() method.

        Labels, bounding boxes: We either cut them to maximal size or fill to fit the maximum size. Returned mask
        tells us, which values are valid.

        :param image: 3-D Tensor of shape [height, width, channels]
        :param labels: 1-D Tensor with labels for every object
        :param bboxes: 2-D Tensor of shape (objects, 4) containing bounding boxes in format [ymin, xmin, ymin, xmax]
                       in relative coordinates
        :param image_id: Id of image, requirement for coco evaluation
        :return: (image, bboxes, labels, mask)
        """
        labels = labels[0 : self.model_config.max_objects]
        bboxes = bboxes[0 : self.model_config.max_objects]
        segmentations = segmentations[0: self.model_config.max_objects]
        bboxes = tf.reshape(bboxes, (-1, 4))  # always keep the second dimension to be 4, even if there are no objects

        # make sure labels and boxes have the correct data type
        labels = tf.cast(labels, dtype=tf.float32)
        bboxes = tf.cast(bboxes, dtype=tf.float32)

        # we resize for the max size to form a batch. Afterwards, we can resize the whole batch
        image_size = (
            self.model_config.image_size + self.model_config.image_size_variation
            if self.train
            else self.model_config.image_size
        )

        height, width = tf.shape(image)[0], tf.shape(image)[1]
        segmentations = tf.image.resize(segmentations, [height, width])
        tf.print(segmentations.shape)

        # make some augmentations
        if self.augmentations:
            if self.resize_before_augmenting:
                additional_space = 1.2  # so that we have something to clip
                pre_resize = tf.cast(image_size, dtype=tf.float32) * additional_space
                ratio = tf.math.minimum(
                    tf.cast(height, dtype=tf.float32) / pre_resize, tf.cast(width, dtype=tf.float32) / pre_resize
                )

                def _preresize():
                    return resize(
                        image,
                        bboxes,
                        segmentations,
                        tf.cast(tf.cast(height, dtype=tf.float32) / ratio, dtype=tf.int32),
                        tf.cast(tf.cast(width, dtype=tf.float32) / ratio, dtype=tf.int32),
                        keep_aspect_ratio=False,
                        random_method=True,
                    )

                # when the image is just slightly better, there is not need to pre-resize
                image, bboxes, segmentations = tf.cond(tf.math.greater(ratio, 1.2), lambda: _preresize(), lambda: (image, bboxes, segmentations))

            # probabilities for random.categorical() are unscaled
            probabilities = [tf.cast(aug.probability, dtype=tf.float32) for aug in self.augmentations]
            selected = tf.random.categorical(tf.math.log([probabilities]), 1, dtype=tf.int32)[0][0]

            # perform augmentation with selected id (nice tf.switch_case() was not working for an unknown reason)
            for idx, aug in enumerate(self.augmentations):
                image, bboxes = tf.cond(selected == idx, lambda: aug.augment(image, bboxes), lambda: (image, bboxes))

        if self.train and not self.model_config.keep_aspect_ratio:
            # randomly chose to keep the image size or spread it out to take the full available space
            image, bboxes, segmentations = self.resize_train(image, bboxes, segmentations, image_size, prob=0.5)
        else:
            # always keep the size or spread depending on the settings
            image, bboxes, segmentations = resize(
                image,
                bboxes,
                segmentations,
                image_size,
                image_size,
                keep_aspect_ratio=self.model_config.keep_aspect_ratio,
                random_method=self.train,
            )

        # calculate mask (one when there is a detected object)
        mask = tf.range(self.model_config.max_objects) < tf.shape(labels)[0]
        mask = tf.cast(mask, dtype=tf.float32)

        # update bounding boxes to the correct shape
        padding_add = tf.math.maximum(self.model_config.max_objects - tf.shape(bboxes)[0], 0)
        bboxes = tf.pad(bboxes, tf.stack([[0, padding_add], [0, 0]]))

        # update labels to correct shape
        labels = tf.pad(labels, tf.stack([[0, padding_add]]))
        labels = tf.cast(labels, dtype=tf.int32)

        # update segmentations
        segmentations = tf.pad(segmentations, tf.stack([[0, padding_add], [0,0], [0,0]]))

        return image, bboxes, labels, mask, image_id, height, width, segmentations

    @tf.function
    def resize_train(self, image, bboxes, segmentations, max_size, prob=0.5):
        image, bboxes, segmentations = tf.cond(
            tf.math.less(tf.random.uniform([], 0.0, 1.0), prob),
            lambda: resize(image, bboxes, segmentations, max_size, max_size, keep_aspect_ratio=True, random_method=True),
            lambda: resize(image, bboxes, segmentations, max_size, max_size, keep_aspect_ratio=False, random_method=True),
        )

        return image, bboxes, segmentations

    @tf.function
    def preprocess_batch(self, images, bboxes, labels, mask, image_ids, heights, widths, segmentations):
        """
        We have the all the inputs in batches, uniformly sized.
        Images/bounding boxes are augmented if this was required.

        First, the images are resized to a random size (from given range) if this is allowed.
        If we use neural network with layers independent on image size, like convolutional ones,
        we could resize the whole batch randomly to further improve our augmentation and prevent overfitting.

        After, inputs for our network are prepared.


        :param image_ids: Id of images, requirement for coco evaluation
        :param heights: original heights of images (not resized)
        :param widths: original widths of images (not resized)
        """
        images = tf.cast(images, tf.float32)

        # select the current batch size, if the variation is greater than 0
        image_size = self.model_config.image_size
        if self.model_config.image_size_variation > 0 and self.train:
            add = tf.random.uniform(
                [],
                minval=-self.model_config.image_size_variation,
                maxval=self.model_config.image_size_variation,
                dtype=tf.int32,
            )

            # TODO 32 depends on network and downsampling
            image_size = ((self.model_config.image_size + add) // 32) * 32

            # resize and pad image to current batch size (aspect ratios are already solved)
            images = random_resize(images, image_size, image_size)
            segmentations = random_resize(segmentations, image_size, image_size)

        # transform bounding boxes from relative to absolute coordinates
        bboxes *= tf.cast(image_size, tf.float32)

        # calculate bounding box properties
        size, local_offset, indices = self.decompose_bounding_boxes(bboxes, image_size, self.model_config.downsample)

        # create heatmap
        bboxes /= self.model_config.downsample
        heatmap_size = image_size // self.model_config.downsample
        heatmap_shape = [tf.shape(images)[0], heatmap_size, heatmap_size, self.model_config.labels]

        if self.model_config.model_type == XModelType.CENTERNET:
            heatmap_dense = tf.numpy_function(func=draw_heatmaps, inp=[heatmap_shape, bboxes, labels], Tout=tf.float32)
            heatmap_dense = tf.reshape(heatmap_dense, heatmap_shape)
            return (
                {"input": images},
                {"heatmap": heatmap_dense, "size": size, "offset": local_offset},
                {
                    "indices": indices,
                    "mask": mask,
                    "bboxes": bboxes,
                    "labels": labels,
                    "ids": image_ids,
                    "heights": heights,
                    "widths": widths,
                },
            )
        else:
            # otherwise we are fittint TTF net
            heatmap_dense, box_target, reg_weight, off, seg_cate, seg_mask = tf.numpy_function(
                func=draw_heatmaps_ttf,
                inp=[heatmap_shape, bboxes, labels, segmentations, tf.constant(True), tf.constant(self.segmentation)],
                Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
            )
            heatmap_dense = tf.reshape(heatmap_dense, heatmap_shape)
            box_target = tf.reshape(box_target, [tf.shape(images)[0], heatmap_size, heatmap_size, 4])
            reg_weight = tf.reshape(reg_weight, [tf.shape(images)[0], heatmap_size, heatmap_size, 1])
            seg_cate = tf.reshape(seg_cate, [tf.shape(images)[0], SOLO_GRID_SIZE, SOLO_GRID_SIZE, heatmap_shape[3]])
            seg_mask = tf.reshape(seg_mask, [tf.shape(images)[0], heatmap_size, heatmap_size, SOLO_GRID_SIZE * SOLO_GRID_SIZE])

            return (
                {"input": images},
                {"heatmap": heatmap_dense, "size": size, "offset": local_offset, "seg_cate": seg_cate, "seg_mask": seg_mask},
                {
                    "indices": indices,
                    "mask": mask,
                    "bboxes": bboxes,
                    "labels": labels,
                    "box_target": box_target,
                    "reg_weight": reg_weight,
                    "ids": image_ids,
                    "heights": heights,
                    "widths": widths,
                },
            )

    @staticmethod
    def decompose_bounding_boxes(bboxes, image_size, downsample):
        # calculate center, size of the bounding box
        center = (bboxes[:, :, 0:2] + bboxes[:, :, 2:4]) / 2.0
        size = -bboxes[:, :, 0:2] + bboxes[:, :, 2:4]

        # downsample center and size
        center = center / float(downsample)
        size = size / float(downsample)

        # calculate point indices so that we can easily get the values from prediction matrices
        center_int = tf.cast(center, dtype=tf.int32)
        heatmap_width = image_size // downsample
        indices = center_int[:, :, 0] * heatmap_width + center_int[:, :, 1]

        # calculate offset
        local_offset = center - tf.cast(center_int, dtype=tf.float32)
        return size, local_offset, indices
