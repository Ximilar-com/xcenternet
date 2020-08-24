import argparse
import numpy as np
import tensorflow as tf

from xcenternet.model.encoder import draw_heatmaps, draw_heatmaps_ttf, bbox_areas_log_np

parser = argparse.ArgumentParser(description="Draw given bounding boxes and their heatmap. All in one B&W image.")
parser.add_argument("--image_size", type=int, default=512, help="image size")
parser.add_argument("--bboxes", type=list, default=[[[92, 208, 327, 355], [70, 158, 300, 305]]], help="bounding boxes")
parser.add_argument("--labels", type=list, default=[[1, 0]])
parser.add_argument("--ttf_version", type=bool, default=False, help="use our tf implementation")
args = parser.parse_args()

heatmap_shape = (1, args.image_size, args.image_size, len(args.labels[0]))

if args.ttf_version:
    heatmap_dense, box_target, reg_weight, _ = draw_heatmaps_ttf(
        heatmap_shape, np.array(args.bboxes), np.array(args.labels)
    )
    heatmap_dense = heatmap_dense[0]
    print("Area weights", [bbox_areas_log_np(np.asarray(box)) for box in args.bboxes[0]])
else:
    heatmap_dense = draw_heatmaps(heatmap_shape, args.bboxes, args.labels)[0]

heatmap_dense = tf.constant(heatmap_dense)
heatmap_dense = tf.image.grayscale_to_rgb(tf.expand_dims(tf.math.reduce_sum(heatmap_dense, 2), 2))
heatmap_dense = heatmap_dense * 255
colors = [[255, 255, 255]] * len(args.bboxes)
image = tf.image.draw_bounding_boxes([heatmap_dense], [np.array(args.bboxes[0]) / float(args.image_size)], colors)

image = tf.cast(image[0], dtype=tf.uint8)
image = tf.image.encode_png(image)
tf.io.write_file("heatmaps.png", image)
