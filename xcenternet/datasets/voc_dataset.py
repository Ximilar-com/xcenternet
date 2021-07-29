import tensorflow_datasets as tfds

from xcenternet.datasets.dataset import Dataset


class VocDataset(Dataset):
    """
    https://www.tensorflow.org/datasets/catalog/voc
    """

    def __init__(self, init_lr):
        super().__init__(20, init_lr)
        self.info = None

    def load_train_datasets(self):
        dataset_train, tinfo = self._load_dataset(name="voc/2007", split="train")
        dataset_val, vinfo = self._load_dataset(name="voc/2007", split="validation")
        dataset_train_2, t2info = self._load_dataset(name="voc/2012", split="train")
        dataset_val_2, v2info = self._load_dataset(name="voc/2012", split="validation")
        self.info = tinfo
        train_dataset = dataset_train.concatenate(dataset_train_2).concatenate(dataset_val_2).concatenate(dataset_val)
        examples = (
            tinfo.splits["train"].num_examples
            + t2info.splits["train"].num_examples
            + v2info.splits["validation"].num_examples
            + vinfo.splits["validation"].num_examples
        )

        return train_dataset, examples

    def load_validation_datasets(self):
        dataset_validation, val_info = self._load_dataset(name="voc/2007", split="test", shuffle_files=False)
        examples = val_info.splits["test"].num_examples
        self.info = val_info

        return dataset_validation, examples

    def _load_dataset(self, name, split, shuffle_files=True):
        dataset, info = tfds.load(
            name=name,
            split=split,
            shuffle_files=shuffle_files,
            with_info=True,
            decoders={"image": tfds.decode.SkipDecoding(),},  # image won't be decoded here
        )
        return dataset, info

    def decode(self, item):
        objects = item["objects"]
        image = self.info.features["image"].decode_example(item["image"])
        labels = objects["label"]
        bboxes = objects["bbox"]
        image_id = 0
        return image, labels, bboxes, [], image_id

    def scheduler(self, epoch):
        if epoch < 100:
            return self.initial_learning_rate
        elif epoch < 140:
            return self.initial_learning_rate * 0.5
        elif epoch < 180:
            return self.initial_learning_rate * 0.1
        else:
            return self.initial_learning_rate * 0.01
