import random
from enum import Enum, auto


class XModelBackbone(Enum):
    RESNET18 = auto()
    RESNET50 = auto()
    EFFICIENTNETB0 = auto()
    MOBILENETV2_10 = auto()
    MOBILENETV2_035 = auto()

    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"


class XModelType(Enum):
    CENTERNET = auto()
    TTFNET = auto()

    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"


class XModelMode(Enum):
    SIMPLE = auto()
    CONCAT = auto()
    SUM = auto()
    DCN = auto()
    DCNSHORTCUT = auto()
    DCNSHORTCUTCONCAT = auto()

    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"


class ModelConfig(object):
    def __init__(
        self,
        size,
        labels,
        max_objects,
        size_variation=0,
        downsample=4,
        keep_aspect_ratio=False,
        model_type=XModelType.CENTERNET,
    ):
        self._size = size
        self._size_variation = size_variation
        self._labels = labels
        self._max_objects = max_objects
        self._downsample = downsample
        self._keep_aspect_ratio = keep_aspect_ratio
        self._debug_class_colors = None
        self._model_type = model_type

    @property
    def image_size(self):
        return self._size

    @property
    def model_type(self):
        return self._model_type

    @property
    def image_size_variation(self):
        return self._size_variation

    @property
    def labels(self):
        return self._labels

    @property
    def max_objects(self):
        return self._max_objects

    @property
    def downsample(self):
        return self._downsample

    @property
    def keep_aspect_ratio(self):
        return self._keep_aspect_ratio

    @property
    def debug_class_colors(self):
        if self._debug_class_colors is None:
            self._debug_class_colors = [
                [float(random.randint(0, 255)), float(random.randint(0, 255)), float(random.randint(0, 255))]
                for _color in range(self.labels)
            ]

        return self._debug_class_colors
