import random

import numpy as np
import json
import cv2
import tensorflow as tf

from xcenternet.datasets.dataset import Dataset
from xcenternet.model.constants import BASE_SEG_MASK_SIZE


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
            image["id"]: {
                "image_id": image["id"],
                "bboxes": [],
                "labels": [],
                "file_name": self.dataset_prefix + image["file_name"],
                "segmentations": [],
                "width": image["width"],
                "height": image["height"]
            } for image in images
        }

        print(self.dataset_prefix + images[0]["file_name"])
        print("LOAD")
        for annotation in annotations:
            records[annotation["image_id"]]["bboxes"].append(annotation["bbox"])
            records[annotation["image_id"]]["labels"].append(annotation["category_id"])
            records[annotation["image_id"]]["segmentations"].append(annotation["segmentation"])

        return list(records.values())

    def _preprocess_record(self, records):
        def gen():
            for record in records:
                bboxes = [
                    [float(bbox[1]), float(bbox[0]), float(bbox[1]) + float(bbox[3]), float(bbox[0]) + float(bbox[2])]
                    for bbox in record["bboxes"]
                ]
                # print(record["segmentations"])
                yield int(record["image_id"]), record["file_name"], [self.labels[label] for label in record["labels"]], bboxes, tf.ragged.constant(record["segmentations"], dtype=tf.float32)

        output_signature = (
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            #tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.RaggedTensorSpec(shape=(None, None, None), dtype=tf.float32)
            # tf.TensorSpec(shape=(None, BASE_SEG_MASK_SIZE, BASE_SEG_MASK_SIZE, 1), dtype=tf.float32)
        )
        return tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    def load_train_datasets(self):
        return self._preprocess_record(self.records_train), len(self.records_train)

    def load_validation_datasets(self):
        return self._preprocess_record(self.records_validation), len(self.records_validation)

    def decode(self, image_id, file_name, labels, bboxes, seg_masks):
        seg_masks = seg_masks.to_tensor()
        image = self._load_image({"file_name": file_name})

        h, w = tf.cast(tf.shape(image)[0], tf.float32), tf.cast(tf.shape(image)[1], tf.float32)
        bboxes /= tf.stack([h, w, h, w])

        # seg_masks = tf_py_segmentations(seg_masks, h, w)
        return image, labels, bboxes, seg_masks, image_id

    def scheduler(self, epoch):
        if epoch < 40:
            return self.initial_learning_rate
        elif epoch < 80:
            return self.initial_learning_rate * 0.1
        else:
            return self.initial_learning_rate * 0.01


def get_segmentations(segments, h, w):
    seg_masks = []
    for segmentations in segments:
        mask = np.zeros((BASE_SEG_MASK_SIZE, BASE_SEG_MASK_SIZE, 1))
        for segmentation in segmentations:
            points = np.asarray([[int((x/w) * BASE_SEG_MASK_SIZE), int((y/h) * BASE_SEG_MASK_SIZE)] for x, y in zip(segmentation[::2], list(reversed(segmentation[::-2])))])
            cv2.fillPoly(mask, [points], color=(1,1,1))
        seg_masks.append(mask)#.astype(np.float32).tolist())
    return np.asarray(seg_masks, dtype=np.float32)


@tf.function
def tf_py_segmentations(segmentations, h, w):
    segmentations = tf.py_function(get_segmentations, [segmentations, h, w], Tout=tf.float32)
    segmentations.set_shape((None,BASE_SEG_MASK_SIZE, BASE_SEG_MASK_SIZE, 1))
    return segmentations


if __name__ == "__main__":
    custom_dataset = CustomDataset("", 0.0)
    custom_dataset.load_train_datasets()
