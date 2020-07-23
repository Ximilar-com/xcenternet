import random

import os
import tensorflow as tf

from xcenternet.datasets.dataset import Dataset
from ximilar.client.utils.json_data import read_json_file_list


class XimilarDataset(Dataset):
    def __init__(self, dataset_path, init_lr):
        labels = read_json_file_list(os.path.join(dataset_path, "labels.json"))
        self.labels = {label["id"]: key for key, label in enumerate(labels)}

        records = read_json_file_list(os.path.join(dataset_path, "images.json"))

        # TODO random split might not distribute labels correctly
        random.seed(2020)
        random.shuffle(records)
        train_num = int(len(records) * 0.8)
        self.records_train = records[:train_num]
        self.records_validation = records[train_num:]

        super().__init__(len(self.labels), init_lr)

    @tf.function
    def _load_image(self, img_path):
        image_encoded = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image_encoded)
        return image

    def _preprocess_record(self, records):
        files, labels, bboxes = [], [], []
        for rec in records:
            b = [obj["data"] for obj in rec["objects"]]
            b = [[bbox[1], bbox[0], bbox[3], bbox[2]] for bbox in b]

            files.append(rec["_file"])
            labels.append([self.labels[obj["detection_label"]] for obj in rec["objects"]])
            bboxes.append(b)

        data = (files, tf.ragged.constant(labels), tf.ragged.constant(bboxes))
        return tf.data.Dataset.from_tensor_slices(data)

    def load_train_datasets(self):
        return self._preprocess_record(self.records_train), len(self.records_train)

    def load_validation_datasets(self):
        return self._preprocess_record(self.records_validation), len(self.records_validation)

    def decode(self, image_path, labels, bboxes):
        image = self._load_image(image_path)
        h, w = tf.shape(image)[0], tf.shape(image)[1]
        bboxes = bboxes.to_tensor()
        bboxes /= tf.stack([h, w, h, w])

        return image, labels, bboxes

    def scheduler(self, epoch):
        if epoch < 40:
            return self.initial_learning_rate
        elif epoch < 80:
            return self.initial_learning_rate * 0.1
        else:
            return self.initial_learning_rate * 0.01
