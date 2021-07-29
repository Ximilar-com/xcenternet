from xcenternet.datasets.dataset import Dataset
import tensorflow_datasets as tfds


class CocoDataset(Dataset):
    def __init__(self, init_lr):
        super().__init__(80, init_lr)
        self.info = None

    def load_train_datasets(self):
        dataset_train, tinfo = self._load_dataset(name="coco/2017", split="train")
        self.info = tinfo

        return dataset_train, tinfo.splits["train"].num_examples

    def load_validation_datasets(self):
        dataset_validation, val_info = self._load_dataset(name="coco/2017", split="validation", shuffle_files=False)
        if self.info is None:
            self.info = val_info
        return dataset_validation, val_info.splits["validation"].num_examples

    def _load_dataset(self, name, split, shuffle_files=True):
        dataset, info = tfds.load(
            name=name,
            split=split,
            shuffle_files=shuffle_files,
            with_info=True,
            data_dir="/data/datasets/mscoco/",
            decoders={"image": tfds.decode.SkipDecoding(),},  # image won't be decoded here
        )
        return dataset, info

    def decode(self, item):
        objects = item["objects"]
        image = self.info.features["image"].decode_example(item["image"])
        labels = objects["label"]
        bboxes = objects["bbox"]
        image_id = item["image/id"]
        return image, labels, bboxes, [], image_id

    def scheduler(self, epoch):
        if epoch < 50:
            return self.initial_learning_rate
        elif epoch < 70:
            return self.initial_learning_rate * 0.5
        elif epoch < 90:
            return self.initial_learning_rate * 0.1
        else:
            return self.initial_learning_rate * 0.01
