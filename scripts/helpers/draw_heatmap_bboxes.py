import argparse
import numpy as np
import tensorflow as tf

from xcenternet.model.encoder import draw_heatmaps, draw_heatmaps_ttf, bbox_areas_log_np

parser = argparse.ArgumentParser(description="Draw given bounding boxes and their heatmap. All in one B&W image.")
parser.add_argument("--image_size", type=int, default=1024, help="image size")
# parser.add_argument("--bboxes", type=list, default=[[[92, 208, 327, 355], [70, 158, 300, 305]]], help="bounding boxes")
# parser.add_argument("--bboxes", type=list, default=[[[ 238, 290, 465, 988 ], [ 38, 290, 665, 988 ]]], help="bounding boxes")
parser.add_argument("--bboxes", type=list, default=[[[ 304, 582, 422, 705 ], [20, 0, 25, 1], [ 304, 582, 422, 705 ], [ 238, 290, 465, 988 ], [ 38, 290, 665, 988 ], [ 38, 490, 665, 788 ], [1000,1000, 1024, 1024]]], help="bounding boxes")
# parser.add_argument("--bboxes", type=list, default=[[[133, 457, 540, 953],[323, 502, 396, 689],[221, 695, 471, 1023], [319, 488, 390, 557]]], help="bounding boxes")
# parser.add_argument("--bboxes", type=list, default=[[[38, 290, 665, 988], [304, 582, 422, 705]]], help="bounding boxes")
parser.add_argument("--labels", type=list, default=[[1, 0]])
parser.add_argument("--ttf_version", type=bool, default=False, help="use our tf implementation")
parser.add_argument("--output", type=str, default="heatmap.png", help="path to heatmap output")
parser.add_argument("--fix_collisions", type=bool, default=False, help="do we want to fix collisions by moving centers?")
args = parser.parse_args()

heatmap_shape = (1, args.image_size, args.image_size, len(args.labels[0]))
labels = np.array([[0 for i in range(len(args.bboxes[0]))]])
if args.ttf_version:
    heatmap, box_target, reg_weight, _, seg_map = draw_heatmaps_ttf(
        heatmap_shape, np.array(args.bboxes), labels,
        fix_collisions=tf.constant(args.fix_collisions)
    )
    heatmap = heatmap[0]
    print("Area weights", [bbox_areas_log_np(np.asarray(box)) for box in args.bboxes[0]])
else:
    heatmap = draw_heatmaps(heatmap_shape, args.bboxes, args.labels)[0]

# print(seg_map[:, :, :, 0])
heatmap_dense = tf.constant(heatmap)
heatmap_dense = tf.image.grayscale_to_rgb(tf.expand_dims(tf.clip_by_value(tf.math.reduce_sum(heatmap_dense, 2), 0.0, 1.0), 2))
heatmap_dense = heatmap_dense * 255
colors = [[255, 255, 255]] * len(args.bboxes)
image = tf.image.draw_bounding_boxes([heatmap_dense], [np.array(args.bboxes[0]) / float(args.image_size)], colors)
image = tf.cast(image[0], dtype=tf.uint8)
image = tf.image.encode_png(image)
tf.io.write_file("heatmap.jpg", image)



