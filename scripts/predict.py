import random

import argparse
import cv2
import numpy as np
import tensorflow as tf
import time

from xcenternet.model.config import ModelConfig, XModelBackbone, XModelMode, XModelType
from xcenternet.model.model_factory import create_model

parser = argparse.ArgumentParser(description="Run predict endpoint of centernet on one image.")
parser.add_argument("--image_size", type=int, default=512, help="image size")
parser.add_argument("--image_path", type=str, default="", help="image path")
parser.add_argument("--batch_size", type=int, default=16, help="size of batch size")
parser.add_argument("--model_type", type=str, default="centernet", help="centernet or ttfnet")
parser.add_argument("--model_mode", type=str, default="concat", help="concat, sum or simple")
parser.add_argument("--backbone", type=str, default="resnet18", help="resnet18, resnet50 or efficientnetb0")
parser.add_argument("--load_model", type=str, default="", help="path to load trained model")
parser.add_argument("--classes", type=int, default=20, help="default number of classes")
parser.add_argument("--max_objects", type=int, default=50, help="default number of classes")
parser.add_argument(
    "--iou_threshold", type=int, default=0.6, help="for deciding whether boxes overlap too much with respect to IOU"
)
parser.add_argument("--score_threshold", type=int, default=0.1, help="when to remove boxes based on score.")
args = parser.parse_args()

# image setup
image_size = args.image_size
labels = args.classes
max_objects = args.max_objects
config = ModelConfig(image_size, labels, max_objects)
batch_size = args.batch_size
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(labels)]

backbone = XModelBackbone[args.backbone.upper()]
model_type = XModelType[args.model_type.upper()]
mode = XModelMode[args.model_mode.upper()]

# load saved model
model = create_model(config.image_size, config.labels, backbone=backbone, model_type=model_type, mode=mode)
if args.load_model:
    model.load_weights(args.load_model)
else:
    print("No model was loaded! Please specify your saved model!")

model.summary()
model.compile()

# create prediction model
predictions = model.outputs
decoded = model.decode(predictions, relative=False, k=max_objects)
pred_model = tf.keras.Model(inputs=model.input, outputs=decoded, name="prediction")
model.trainable = False
pred_model.trainable = False

# load image
image = cv2.imread(args.image_path)
image = cv2.resize(image, (args.image_size, args.image_size))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

batch = np.asarray([image for i in range(1)])

# first inference
result = pred_model.predict(batch)

# test the time
start_time = time.time()
for i in range(20):
    pred_model.predict(batch)
print("Average speed per one image (20 runs): ", (time.time() - start_time) / 20.0)

# decode it
start_time = time.time()
dec = result[0, :, :]
dec_mask = dec[:, 4] >= 0.3
result = tf.boolean_mask(dec, dec_mask)
bboxes = result[:, 0:4] * 4.0
bboxes = tf.clip_by_value(tf.cast(bboxes, tf.float32), 0.0, float(image_size - 1.0))
labels = tf.cast(result[:, 5], tf.int32)
scores = result[:, 4]

# the original nms is not enough lets apply the stronger one
print(bboxes.shape, scores.shape)
selected_indices = tf.image.non_max_suppression(
    bboxes, scores, max_objects, iou_threshold=args.iou_threshold, score_threshold=args.score_threshold
)
selected_boxes = tf.gather(bboxes, selected_indices).numpy()
selected_labels = tf.gather(labels, selected_indices).numpy()
selected_scores = tf.gather(scores, selected_indices).numpy()
print("Time to decode: ", (time.time() - start_time))

# print the bounding box
for box, label, score in zip(selected_boxes, selected_labels, selected_scores):
    box_int = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
    print(f"Found box {box_int} with label {label} and probability {score:.2f}")
    cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), colors[label], 2)

image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
file = "result.jpg"
cv2.imwrite(file, image)

print()
print(f"Image with bounding boxes saved to {file} file.")
