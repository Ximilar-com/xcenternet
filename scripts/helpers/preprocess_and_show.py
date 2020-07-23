import matplotlib.pyplot as plt
import tensorflow as tf

from xcenternet.datasets.voc_dataset import VocDataset
from xcenternet.model.config import ModelConfig
from xcenternet.model.encoder import draw_heatmaps as draw_heatmaps
from xcenternet.model.preprocessing.augmentations import HardAugmentation, EasyAugmentation
from xcenternet.model.preprocessing.batch_preprocessing import BatchPreprocessing
from xcenternet.tensorboard.visualization import draw_bounding_boxes

#
# Preprocess couple of images and show the result.
#

# Setup script.
SHOW_BATCH_IMAGES = 10
SIZE = 512

# setup augmentations
hard_augmentation = HardAugmentation(0.9)
easy_augmentation = EasyAugmentation(0.1)

# Setup dataset.
dataset = VocDataset(0.0)
train_dataset, _ = dataset.load_train_datasets()
config = ModelConfig(SIZE, 20, 10, size_variation=128)
train_processing = BatchPreprocessing(config, train=True, augmentations=[hard_augmentation, easy_augmentation])

# Get one batch.
dataset_train = (
    train_dataset.take(SHOW_BATCH_IMAGES)
    .map(dataset.decode)
    .map(train_processing.prepare_for_batch)
    .batch(SHOW_BATCH_IMAGES)
    .map(train_processing.preprocess_batch)
)
ds = iter(dataset_train)
examples = next(ds)

# Extract data from example.
images = examples[0]["input"]
bounding_box_sizes, local_offsets = examples[1]["size"], examples[1]["offset"]
indices, labels = examples[2]["indices"], examples[2]["labels"]

# Calculate bounding boxes.
downsample = 4
width = tf.cast(tf.shape(images)[2] // downsample, dtype=tf.float32)
center = tf.tile(tf.expand_dims(tf.cast(indices, dtype=tf.float32), -1), [1, 1, 2])
center = tf.map_fn(lambda c: tf.map_fn(lambda i: tf.stack([i[0] // width, i[1] % width]), c), center)
center = (center + local_offsets) * 4.0
bboxes = tf.tile(center, [1, 1, 2])
sizes = (tf.tile(bounding_box_sizes, [1, 1, 2]) / 2) * tf.constant([-1.0, -1.0, 1.0, 1.0]) * downsample
bboxes = (bboxes + sizes) / tf.cast(
    (tf.stack([tf.shape(images)[1], tf.shape(images)[2], tf.shape(images)[1], tf.shape(images)[2]])), dtype=tf.float32
)

# Convert image to 0 - 1.
images = tf.cast(images, dtype=tf.float32)

# Create figure and axes.
for i in range(len(images)):
    image = images[i]
    image = draw_bounding_boxes(image, bboxes[i], labels[i], config)
    image = image / 255.0

    fig, ax = plt.subplots(1)
    ax.imshow(image)
    plt.show()
