import random
import json
import os
import argparse
import progressbar
import numpy as np
import tensorflow as tf
from xcenternet.model.centernet import XTTFModel, XCenternetModel

from xcenternet.datasets import CocoDataset, VocDataset, CustomDataset
from xcenternet.model.model_factory import create_model
from xcenternet.model.preprocessing.batch_preprocessing import BatchPreprocessing
from xcenternet.model.config import ModelConfig, XModelType, XModelBackbone, XModelMode
from xcenternet.model.evaluation.mean_average_precision import MAP

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

assert callable(progressbar.progressbar), "Using wrong progressbar module, install 'progressbar2' instead."


def load_coco_classes(coco):
    """ Loads the class to label mapping (and inverse) for COCO.
    """
    # load class names (name -> label)
    categories = coco.loadCats(coco.getCatIds())
    categories.sort(key=lambda x: x["id"])

    classes = {}
    coco_labels = {}
    coco_labels_inverse = {}
    for c in categories:
        coco_labels[len(classes)] = c["id"]
        coco_labels_inverse[c["id"]] = len(classes)
        classes[c["name"]] = len(classes)

    return classes, coco_labels, coco_labels_inverse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation of centernet.")
    parser.add_argument("--dataset", type=str, default="voc", help="voc or coco, custom (coco format)")
    parser.add_argument("--dataset_path", type=str, default="", help="path to custom dataset")
    parser.add_argument("--model_type", type=str, default="centernet", help="centernet or ttfnet")
    parser.add_argument("--model_mode", type=str, default="dcnshortcut", help="concat, sum or simple")
    parser.add_argument(
        "--no_decode", dest="no_decode", action="store_true", help="do not apply decode to model output"
    )
    parser.add_argument("--backbone", type=str, default="resnet18", help="resnet18, resnet50 or efficientnetb0")
    parser.add_argument("--image_size", type=int, default=512, help="image size")
    parser.add_argument("--load_model", type=str, default="", help="path to load trained model")
    parser.add_argument("--load_weights", type=str, default="", help="path to load trained model weights")
    parser.add_argument("--threshold", type=float, default=0.3, help="float, prob threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="float, iou threshold")
    parser.add_argument("--split_set", type=str, default="validation", help="train or validation (default)")
    parser.add_argument("--batch_size", type=int, default=16, help="default batch size (default 16)")
    parser.add_argument("--max_objects", type=int, default=100, help="max objects to detect (default 100)")
    parser.add_argument(
        "--evaluate_one_by_one",
        dest="evaluate_one_by_one",
        action="store_true",
        help="accumulate results image by image",
    )
    args = parser.parse_args()

    backbone = XModelBackbone[args.backbone.upper()]
    model_type = XModelType[args.model_type.upper()]
    mode = XModelMode[args.model_mode.upper()]

    # dataset
    dataset = None
    if args.dataset == "voc":
        dataset = VocDataset(0)
    elif args.dataset == "coco":
        dataset = CocoDataset(0)
        coco_true = COCO(os.path.join("annotations", "instances_val2017.json"))
        _, coco_labels, coco_labels_inverse = load_coco_classes(coco_true)
    elif args.dataset == "custom":
        dataset = CustomDataset(args.dataset_path, args.dataset_path, 0)
        coco_true = COCO(args.dataset_path)
        _, coco_labels, coco_labels_inverse = load_coco_classes(coco_true)
    else:
        print(f"Unknown dataset {args.dataset}.")
        exit()

    labels = dataset.classes

    # image setup
    image_size = args.image_size
    max_objects = args.max_objects
    config = ModelConfig(image_size, labels, max_objects, model_type=model_type)
    batch_size = args.batch_size
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(labels)]

    # validation dataset
    if args.split_set == "validation":
        eval_dataset, eval_examples = dataset.load_validation_datasets()
    else:
        eval_dataset, eval_examples = dataset.load_train_datasets()

    autotune = None
    validation_processing = BatchPreprocessing(config, train=False)
    ds = (
        eval_dataset.map(dataset.decode)
        .map(validation_processing.prepare_for_batch, num_parallel_calls=autotune)
        .batch(batch_size)
        .map(validation_processing.preprocess_batch, num_parallel_calls=autotune)
    )

    model_class = XTTFModel if model_type == XModelType.TTFNET else XCenternetModel

    if args.load_model:
        model = tf.keras.models.load_model(args.load_model, custom_objects={model_class.__name__: model_class})
    else:
        model = create_model(
            config.image_size,
            config.labels,
            pretrained_backbone=False,
            backbone=backbone,
            model_type=model_type,
            mode=mode,
        )
        model.load_weights(args.load_weights)
    model.summary()

    # compiled should run faster
    @tf.function
    def decode(predictions):
        return model_class.decode(model, predictions, relative=False, k=max_objects)

    mean_average_precision = MAP(labels, iou_threshold=args.iou_threshold, score_threshold=args.threshold)

    batches = tf.data.experimental.cardinality(ds).numpy()
    if batches < 0:
        batches = int(eval_examples / args.batch_size)

    result_ids = []

    for _, batch in zip(progressbar.progressbar(range(batches), prefix="Evaluating: "), ds):
        batch_predict = model.predict(batch[0])
        if not args.no_decode:
            batch_predict = decode(batch_predict)

        batch_images, target_input, targets = batch

        batch_ids = targets["ids"]
        batch_images = batch_images["input"]
        batch_masks = targets["mask"]
        batch_bboxes = targets["bboxes"]
        batch_labels = targets["labels"]
        batch_heights = targets["heights"]
        batch_widths = targets["widths"]
        image_size = float(args.image_size / 4.0)

        if args.evaluate_one_by_one:
            image_ids = []
            batch_ids = batch_ids if batch_ids is not None else tf.zeros((batch_size,))
            for image, masks, bboxes, labels, predictions, id_image, height, width in zip(
                batch_images,
                batch_masks,
                batch_bboxes,
                batch_labels,
                batch_predict,
                batch_ids,
                batch_heights,
                batch_widths,
            ):
                masks = masks.numpy().astype(np.bool)
                bboxes = bboxes.numpy()[masks].astype(np.float64)
                labels = labels.numpy()[masks].astype(np.int32)
                height = height.numpy().astype(np.float32)
                width = width.numpy().astype(np.float32)

                # for pascal eval
                mean_average_precision.update_state(predictions, bboxes, labels)

                # for coco eval
                if args.dataset in ["coco", "custom"]:
                    result = mean_average_precision._get_detections(
                        predictions, labels, args.threshold, allresult=False
                    )
                    image_ids.append(int(id_image.numpy()))

                    for z in range(len(result)):
                        score = result[z][4]
                        label = int(result[z][5])
                        box = (result[z][0:4]).tolist()
                        box = [
                            box[1] / image_size,
                            box[0] / image_size,
                            box[3] / image_size,
                            box[2] / image_size,
                        ]  # convert to relative coordinates
                        box = [
                            box[0] * width,
                            box[1] * height,
                            (box[2] - box[0]) * width,
                            (box[3] - box[1]) * height,
                        ]  # rescaled bbox for original image size and coco format

                        category_id = coco_labels[label] if args.dataset == "coco" else label
                        image_result = {
                            "image_id": int(id_image.numpy()),
                            "category_id": category_id,
                            "score": float(score),
                            "bbox": box,
                        }
                        result_ids.append(image_result)
        else:
            mean_average_precision.update_state_batch(batch_predict, batch_bboxes, batch_labels, batch_masks)

    if args.dataset in ["coco", "custom"]:
        json.dump(result_ids, open("result_bbox_results.json", "w"), indent=4)
        coco_pred = coco_true.loadRes("result_bbox_results.json")
        coco_eval = COCOeval(coco_true, coco_pred, "bbox")
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        print(coco_eval.stats)

    result = mean_average_precision.result()
    detail_print = "\n".join(
        [f"label {label}: {val[0]:0.2f} mAP ({val[1]:.0f} annotations)" for label, val in result["per_class"].items()]
    )

    print("------")
    print("------")
    print(f"mAP total (overall, per class: {result['overall']:0.2f}")
    print(f"mAP total (weighted per annotation num): {result['weighted']:0.2f}")
    print("------")
    print(detail_print)
    print("finish")
