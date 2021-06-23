import argparse
import numpy as np
import tensorflow as tf

from xcenternet.model.encoder import draw_heatmaps, draw_heatmaps_ttf, draw_heatmaps_ttf2, bbox_areas_log_np

parser = argparse.ArgumentParser(description="Draw given bounding boxes and their heatmap. All in one B&W image.")
parser.add_argument("--image_size", type=int, default=1024, help="image size")
parser.add_argument("--bboxes", type=list, default=[[[ 304, 582, 422, 705 ], [ 304, 582, 422, 705 ], [ 238, 290, 465, 988 ], [ 38, 290, 665, 988 ], [ 38, 490, 665, 788 ]]], help="bounding boxes")
parser.add_argument("--labels", type=list, default=[[1, 1, 0,0,0]])
parser.add_argument("--ttf_version", type=bool, default=False, help="use our tf implementation")
parser.add_argument("--output", type=str, default="heatmap.png", help="path to heatmap output")
args = parser.parse_args()

heatmap_shape = (1, args.image_size, args.image_size, len(args.labels[0]))

if args.ttf_version:
    heatmap, box_target, reg_weight, _ = draw_heatmaps_ttf2(
        heatmap_shape, np.array(args.bboxes), np.array(args.labels)
    )
    heatmap = heatmap[0]
    print("Area weights", [bbox_areas_log_np(np.asarray(box)) for box in args.bboxes[0]])
else:
    heatmap = draw_heatmaps(heatmap_shape, args.bboxes, args.labels)[0]

heatmap_dense = tf.constant(heatmap)
heatmap_dense = tf.image.grayscale_to_rgb(tf.expand_dims(tf.clip_by_value(tf.math.reduce_sum(heatmap_dense, 2), 0.0, 1.0), 2))
heatmap_dense = heatmap_dense * 255
colors = [[255, 255, 255]] * len(args.bboxes)
image = tf.image.draw_bounding_boxes([heatmap_dense], [np.array(args.bboxes[0]) / float(args.image_size)], colors)
image = tf.cast(image[0], dtype=tf.uint8)
image = tf.image.encode_png(image)
tf.io.write_file("heatmap_full.jpg", image)



