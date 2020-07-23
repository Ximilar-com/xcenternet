import tensorflow as tf
from abc import ABC, abstractmethod


class Dataset(ABC):
    """
    This base class for designed to make use of out training script a bit easier.
    Just implement the provided methods and you are ready to go.
    """

    def __init__(self, classes: int, init_lr: float):
        self._classes = classes
        self.initial_learning_rate = init_lr

    def scheduler(self, epoch: int) -> float:
        """
        Epoch lr rate scheduler for your dataset
        """
        raise NotImplementedError

    @property
    def classes(self) -> int:
        """
        :return: number of classes in this dataset
        """
        return self._classes

    @abstractmethod
    def load_train_datasets(self) -> (tf.data.Dataset, float):
        """
        :return: training dataset(tf.data.Dataset), size
        """
        pass

    @abstractmethod
    def load_validation_datasets(self) -> (tf.data.Dataset, float):
        """
        :return: validation dataset (tf.data.Dataset), size
        """
        pass

    @abstractmethod
    def decode(self, *args, **kwargs):
        """
        :return: (image, labels, bboxes) where
        """
        pass
