"""
This script is modied version of script from https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3 repository.
"""

import cv2
import json
import numpy as np
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow_datasets as tfds


def compute_iou(box1, box2):
    # xmin, ymin, xmax, ymax
    A1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    A2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    xmin = max(box1[0], box2[0])
    ymin = max(box1[1], box2[1])
    xmax = min(box1[2], box2[2])
    ymax = min(box1[3], box2[3])

    if ymin >= ymax or xmin >= xmax:
        return 0
    return ((xmax - xmin) * (ymax - ymin)) / (A1 + A2)


def make_image(data, image, id, ratio=1):
    blank = data[0]
    boxes = data[1]
    label = data[2]

    image = 255 - image
    image = cv2.resize(image, (int(28 * ratio), int(28 * ratio)))
    h, w = image.shape

    while True:
        xmin = np.random.randint(0, SIZE - w, 1)[0]
        ymin = np.random.randint(0, SIZE - h, 1)[0]
        xmax = xmin + w
        ymax = ymin + h
        box = [xmin, ymin, xmax, ymax]

        iou = [compute_iou(box, b) for b in boxes]
        if max(iou) < 0.02:
            boxes.append(box)
            label.append(id)
            break

    for i in range(w):
        for j in range(h):
            x = xmin + i
            y = ymin + j
            blank[y][x] = image[j][i]

    return blank


SIZE = 416
image_sizes = [3, 6, 3]  # small, medium, big
train_ds, test_ds = tfds.load("mnist", split=["train", "test"])

datasets = {"train": {"ds": train_ds, "count": 1000}, "test": {"ds": test_ds, "count": 200}}

for dataset in list(datasets.keys()):
    os.makedirs(dataset, exist_ok=True)
    a = 0
    # create mscoco format
    data_file = {
        "categories": [{"id": i, "name": "number_" + str(i)} for i in range(10)],
        "images": [],
        "annotations": [],
    }

    examples = [example for example in datasets[dataset]["ds"].as_numpy_iterator()]

    for image_num in range(datasets[dataset]["count"]):
        image_path = os.path.realpath(os.path.join(dataset, "%06d.jpg" % (image_num)))

        blanks = np.ones(shape=[SIZE, SIZE, 3]) * 255
        bboxes = [[0, 0, 1, 1]]
        labels = [0]
        data = [blanks, bboxes, labels]
        bboxes_num = 0

        # ratios small, medium, big objects
        ratios = [[0.5, 0.8], [1.0, 1.5, 2.0], [3.0, 4.0]]
        for i in range(len(ratios)):
            N = random.randint(0, image_sizes[i])
            if N != 0:
                bboxes_num += 1
            for _ in range(N):
                ratio = random.choice(ratios[i])
                idx = random.randint(0, len(examples) - 1)
                data[0] = make_image(data, examples[idx]["image"], examples[idx]["label"], ratio)

        if bboxes_num == 0:
            continue
        cv2.imwrite(image_path, data[0])
        data_file["images"].append({"file_name": image_path, "id": image_num})

        for i in range(len(labels)):
            if i == 0:
                continue
            xmin = bboxes[i][0]
            ymin = bboxes[i][1]
            xmax = bboxes[i][2]
            ymax = bboxes[i][3]
            data_file["annotations"].append(
                {
                    "image_id": image_num,
                    "bbox": [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)],
                    "category_id": int(labels[i]),
                    "iscrowd": 0,
                    "area": int(xmax - xmin) * int(ymax - ymin),
                    "id": a,
                }
            )
            a += 1
        print(image_num)

    with open(dataset + ".json", "w") as fp:
        json.dump(data_file, fp, indent=4)
        print("Created", dataset + ".json")
