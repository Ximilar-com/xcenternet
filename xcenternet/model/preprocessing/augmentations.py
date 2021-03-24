import abc

from tf_image.application.augmentation_config import AugmentationConfig, ColorAugmentation, AspectRatioAugmentation
from tf_image.application.tools import random_augmentations
from tf_image.core.bboxes.resize import random_pad_to_square
from tf_image.core.random import random_function_bboxes, random_function
from xcenternet.model.preprocessing.color import tf_py_contrast, tf_py_blur, tf_py_dropout


class Augmentation(metaclass=abc.ABCMeta):
    def __init__(self, probability):
        self.probability = probability

    @abc.abstractmethod
    def augment(self, image, bboxes):
        raise NotImplementedError()


class EasyAugmentation(Augmentation):
    def __init__(self, probability):
        super().__init__(probability)

        self.augmentation_config = AugmentationConfig()
        self.augmentation_config.color = ColorAugmentation.LIGHT
        self.augmentation_config.crop = True
        self.augmentation_config.distort_aspect_ratio = AspectRatioAugmentation.NONE
        self.augmentation_config.quality = True
        self.augmentation_config.erasing = False
        self.augmentation_config.rotate90 = False
        self.augmentation_config.rotate_max = 0
        self.augmentation_config.flip_vertical = False
        self.augmentation_config.flip_horizontal = True
        self.padding_square = False

    def augment(self, image, bboxes):
        return random_augmentations(image, self.augmentation_config, bboxes=bboxes)


class HardAugmentation(Augmentation):
    def __init__(self, probability):
        super().__init__(probability)

        self.augmentation_config = AugmentationConfig()
        self.augmentation_config.color = ColorAugmentation.AGGRESSIVE
        self.augmentation_config.crop = True
        self.augmentation_config.distort_aspect_ratio = AspectRatioAugmentation.TOWARDS_SQUARE
        self.augmentation_config.quality = True
        self.augmentation_config.erasing = True
        self.augmentation_config.rotate90 = False
        self.augmentation_config.rotate_max = 13
        self.augmentation_config.flip_vertical = False
        self.augmentation_config.flip_horizontal = True
        self.padding_square = False

    def augment(self, image, bboxes):
        image, bboxes = random_augmentations(image, self.augmentation_config, bboxes=bboxes)
        image, bboxes = random_function_bboxes(image, bboxes, random_pad_to_square, 0.3)

        # unfortunately we are still missing some augmentations in tf_image
        image = random_function(image, tf_py_contrast, prob=0.3)
        image = random_function(image, tf_py_blur, prob=0.3)
        image = random_function(image, tf_py_dropout, prob=0.3)

        return image, bboxes
