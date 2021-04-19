"""
Original file comes from Fizyr, but it was heavily modified by us. Structure changed, batch processing added.

Copyright 2017-2018 Fizyr (https://fizyr.com)
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import tensorflow as tf
from xcenternet.model.decoder import filter_detections
from xcenternet.model.evaluation.overlap import compute_overlap


class MAP(object):
    def __init__(self, classes, iou_threshold=0.5, score_threshold=0.05, max_size=None):
        """
        Creates object which calculates mAP (mean Average Precision) for our detection predictions.

        :param classes: how many different object classes do we have
        :param iou_threshold: "Intersection over Union" threshold when we consider the object to be correctly detected
        :param score_threshold: probability from which we consider the detection to be valid
        :param max_size: maximum size used to clip the predicted bounding boxes
        """
        self.classes = classes
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

        self.false_positives = None
        self.true_positives = None
        self.scores = None
        self.num_annotations = None
        self.max_size = max_size

        self.reset_states()

    def reset_states(self):
        """
        Resets all the accumulated values. Has to be called before every new epoch if we want to reuse this object.
        """
        self.false_positives = [[] for _ in range(self.classes)]
        self.true_positives = [[] for _ in range(self.classes)]
        self.scores = [[] for _ in range(self.classes)]
        self.num_annotations = [0.0 for _ in range(self.classes)]

    def update_state_batch(self, predictions, bboxes, labels, mask):
        """
        Update our results using given batch predictions and their ground truth values.

        :param predictions:
        :param bboxes:
        :param labels:
        :param mask:
        :return:
        """
        annotations = self._get_batch_annotations(mask, bboxes, labels, self.classes)

        detections = self._get_batch_detections(predictions, self.classes, score_threshold=self.score_threshold)

        self._evaluate_batch(annotations, detections, self.classes, iou_threshold=self.iou_threshold)

    def update_state(self, predictions, bboxes, labels):
        """
        Update our results using given predictions for a single image and its ground truth values.

        :param predictions:
        :param bboxes:
        :param labels:
        :return:
        """
        annotations = self._get_annotations(bboxes, labels, self.classes)

        detections = self._get_detections(predictions, self.classes, score_threshold=self.score_threshold)

        self._evaluate_batch([annotations], [detections], self.classes, iou_threshold=self.iou_threshold)

    def result(self):
        average_precisions = {}
        for label in range(self.classes):
            false_positives_label = np.array(self.false_positives[label])
            true_positives_label = np.array(self.true_positives[label])
            scores_label = np.array(self.scores[label])
            num_annotations_label = self.num_annotations[label]

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations_label == 0:
                average_precisions[label] = 0, 0
                continue

            # sort by score
            indices = np.argsort(-scores_label)
            false_positives_label = false_positives_label[indices]
            true_positives_label = true_positives_label[indices]

            # compute false positives and true positives
            false_positives_label = np.cumsum(false_positives_label)
            true_positives_label = np.cumsum(true_positives_label)

            # compute recall and precision
            recall = true_positives_label / num_annotations_label
            precision = true_positives_label / np.maximum(
                true_positives_label + false_positives_label, np.finfo(np.float64).eps
            )

            # compute average precision
            average_precision = self._compute_ap(recall, precision)
            average_precisions[label] = average_precision, num_annotations_label

        annotations = sum([class_stats[1] for _, class_stats in average_precisions.items()])
        weighted = (
            sum([class_stats[0] * class_stats[1] for _, class_stats in average_precisions.items()]) / annotations
            if annotations > 0
            else 0
        )

        non_empty_classes = sum([class_stats[1] > 0 for _, class_stats in average_precisions.items()])
        overall = (
            sum([class_stats[0] for _, class_stats in average_precisions.items()]) / non_empty_classes
            if non_empty_classes > 0
            else 0
        )

        return {"overall": overall, "weighted": weighted, "per_class": average_precisions}

    def _evaluate_batch(self, annotations_batch, detections_batch, class_num, iou_threshold):
        # process detections and annotations
        for label in range(class_num):
            for detections, annotations in zip(detections_batch, annotations_batch):
                detections, annotations = detections[label], annotations[label]

                self.num_annotations[label] += annotations.shape[0]

                detected_annotations = []
                for d in detections:
                    self.scores[label].append(d[4])

                    if annotations.shape[0] == 0:
                        self.false_positives[label].append(1)
                        self.true_positives[label].append(0)
                        continue

                    overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)[0]
                    assigned_annotation = np.argmax(overlaps)
                    max_overlap = overlaps[assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        self.false_positives[label].append(0)
                        self.true_positives[label].append(1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        self.false_positives[label].append(1)
                        self.true_positives[label].append(0)

    def _compute_ap(self, recall, precision):
        """ Compute the average precision, given the recall and precision curves.
        Code originally from https://github.com/rbgirshick/py-faster-rcnn.
        # Arguments
            recall:    The recall curve (list).
            precision: The precision curve (list).
        # Returns
            The average precision as computed in py-faster-rcnn.
        """
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def _unpack_result(self, result, score_threshold):
        mask = result[:, 4] >= score_threshold
        result = tf.boolean_mask(result, mask)
        bboxes = result[:, 0:4]

        if self.max_size is not None:
            bboxes = tf.clip_by_value(tf.cast(bboxes, tf.float32), 0.0, float(self.max_size))

        labels = tf.cast(result[:, 5], tf.int32)
        scores = result[:, 4]

        max_objects = result.shape[0]
        selected_indices = tf.image.non_max_suppression(bboxes, scores, max_objects, iou_threshold=0.5)
        selected_boxes = tf.gather(bboxes, selected_indices).numpy()
        selected_labels = tf.gather(labels, selected_indices).numpy()
        selected_scores = tf.gather(scores, selected_indices).numpy()

        return selected_boxes, selected_scores, selected_labels

    def _get_batch_detections(self, result, class_num, score_threshold):
        """ Get the detections from the model using the generator.
        The result is a list of lists such that the size is:
            all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
        # Arguments
        # Returns
            A list of lists containing the detections for each image in the generator.
            Bounding boxes are in heatmap coordinates. (As present in the preprocessed dataset and predictions.)
        """
        samples = result.shape[0]
        all_detections = [[None for i in range(class_num)] for j in range(samples)]

        for i in range(samples):
            all_detections[i] = self._get_detections(result[i], class_num, score_threshold)

        return all_detections

    def _get_detections(self, result, class_num, score_threshold, allresult=True):
        results = filter_detections(
            result, score_threshold=score_threshold, nms_iou_threshold=0.5, max_size=self.max_size
        )
        results = results.numpy().astype("double")

        if not allresult:
            return results

        # copy detections to all_detections
        detections = [None for i in range(class_num)]
        for label in range(class_num):
            detections[label] = results[results[:, -1] == label, :-1]

        return detections

    def _get_batch_annotations(self, mask, bboxes, labels, class_num):
        """ Get the ground truth annotations from the generator.
            The result is a list of lists such that the size is:
                all_detections[num_images][num_classes] = annotations[num_detections, 5]
            # Arguments
            # Returns
                A list of lists containing the annotations for each image in the generator.
                Bounding boxes are in heatmap coordinates. (As present in the preprocessed dataset and predictions.)
            """
        mask, bboxes, labels = mask.numpy(), bboxes.numpy(), labels.numpy()
        mask = mask.astype(dtype=np.bool)
        bboxes = bboxes.astype(dtype=np.float64)
        samples = mask.shape[0]

        # copy detections to all_annotations
        all_annotations = [[None for i in range(class_num)] for j in range(samples)]
        for i in range(samples):
            sample_bboxes, sample_labels = bboxes[i][mask[i]], labels[i][mask[i]]
            all_annotations[i] = self._get_annotations(sample_bboxes, sample_labels, class_num)

        return all_annotations

    def _get_annotations(self, bboxes, labels, class_num):
        annotations = [None for _ in range(class_num)]
        for label in range(class_num):
            annotations[label] = bboxes[labels == label, :].copy()

        return annotations
