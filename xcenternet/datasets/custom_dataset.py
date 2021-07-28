import random

import json
import tensorflow as tf

from xcenternet.datasets.dataset import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataset_path_tr, dataset_path_te, init_lr, dataset_prefix):
        self.dataset_prefix = dataset_prefix
        self.records_train, self.labels = self.load_file(dataset_path_tr)
        self.records_validation, _ = self.load_file(dataset_path_te)

        super().__init__(len(self.labels), init_lr)

    def load_file(self, file_path):
        with open(file_path) as f:
            data = json.load(f)

        labels = data["categories"]
        labels = {label["id"]: index for index, label in enumerate(labels)}

        images = data["images"]
        annotations = data["annotations"]

        records = self.load_records(images, annotations)
        random.seed(2020)
        random.shuffle(records)
        return records, labels

    @tf.function
    def _load_image(self, record):
        img_path = record["file_name"]
        image_encoded = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image_encoded)
        return image

    def load_records(self, images, annotations):
        records = {
            image["id"]: {"image_id": image["id"], "bboxes": [], "labels": [], "file_name": self.dataset_prefix + image["file_name"], "segmentations": []}
            for image in images
        }

        print(self.dataset_prefix + images[0]["file_name"])
        print("LOAD")
        for annotation in annotations:
            records[annotation["image_id"]]["bboxes"].append(annotation["bbox"])
            records[annotation["image_id"]]["labels"].append(annotation["category_id"])
            #records[annotation["image_id"]]["segmentations"].append(annotation["segmentation"])

        return list(records.values())

    def _preprocess_record(self, records):
        def gen():
            for record in records:
                bboxes = record["bboxes"]
                bboxes = [
                    [float(bbox[1]), float(bbox[0]), float(bbox[1]) + float(bbox[3]), float(bbox[0]) + float(bbox[2])]
                    for bbox in bboxes
                ]
                yield {
                    "image_id": int(record["image_id"]),
                    "file_name": record["file_name"],
                    "labels": [self.labels[label] for label in record["labels"]],
                    "bboxes": bboxes,
                    #"segmentations": record["segmentations"]
                }

        output_types = {"image_id": tf.int32, "file_name": tf.string, "labels": tf.float32, "bboxes": tf.float32} #, "segmentations": tf.float32}
        output_shapes = {"file_name": (), "labels": (None,), "bboxes": (None, 4), "image_id": ()} #, "segmentations": (None, None,)}

        return tf.data.Dataset.from_generator(gen, output_types=output_types, output_shapes=output_shapes)

    def load_train_datasets(self):
        return self._preprocess_record(self.records_train), len(self.records_train)

    def load_validation_datasets(self):
        return self._preprocess_record(self.records_validation), len(self.records_validation)

    def decode(self, record):
        image = self._load_image(record)
        labels = record["labels"]
        bboxes = record["bboxes"]
        # segmentations = record["segmentations"]
        image_id = record["image_id"]

        h, w = tf.cast(tf.shape(image)[0], tf.float32), tf.cast(tf.shape(image)[1], tf.float32)
        bboxes /= tf.stack([h, w, h, w])

        return image, labels, bboxes, image_id #[] #segmentations

    def scheduler(self, epoch):
        if epoch < 40:
            return self.initial_learning_rate
        elif epoch < 80:
            return self.initial_learning_rate * 0.1
        else:
            return self.initial_learning_rate * 0.01


if __name__ == "__main__":
    custom_dataset = CustomDataset("", 0.0)
    custom_dataset.load_train_datasets()
