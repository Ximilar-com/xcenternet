import argparse
import os
import tensorflow as tf

from xcenternet.datasets import CocoDataset, VocDataset, CustomDataset
from xcenternet.model.callbacks import MAPValidationCallback
from xcenternet.model.config import ModelConfig, XModelType, XModelBackbone, XModelMode
from xcenternet.model.model_factory import create_model, load_and_update_model, load_pretrained_weights
from xcenternet.model.preprocessing.augmentations import EasyAugmentation, HardAugmentation
from xcenternet.model.preprocessing.batch_preprocessing import BatchPreprocessing
from xcenternet.tensorboard.callback import XTensorBoardCallback
from xcenternet.tensorboard.image_log import ImageLog
from xcenternet.tensorboard.result_log import ResultImageLogCallback

parser = argparse.ArgumentParser(description="Run training of centernet on VOC.")
parser.add_argument("--dataset", type=str, default="voc", help="voc, coco, custom (coco format)")
parser.add_argument("--dataset_path_tr", type=str, default="", help="path to the train file")
parser.add_argument("--dataset_path_te", type=str, default="", help="path to the test file")
parser.add_argument("--dataset_prefix", type=str, default="", help="prefix to path to images")
parser.add_argument("--model_type", type=str, default="centernet", help="centernet or ttfnet")
parser.add_argument("--model_mode", type=str, default="dcnshortcut", help="concat, sum or simple")
parser.add_argument("--backbone", type=str, default="resnet18", help="resnet18, resnet50 or efficientnetb0")
parser.add_argument("--pretrained", type=str, default="", help="path to a pretrained model (SavedModel)")
parser.add_argument("--pretrained_weights", type=str, default="", help="path to a pretrained weights (h5)")
parser.add_argument(
    "--random_weights", dest="random_weights", action="store_true", help="do not start with imagenet weights"
)
parser.add_argument("--epochs", type=int, default=101, help="number of epochs to train")
parser.add_argument("--image_size", type=int, default=512, help="image size")
parser.add_argument(
    "--size_variation", type=int, default=128, help="multi scale training (image_size +- size_variation)"
)
parser.add_argument("--batch_size", type=int, default=28, help="size of batch size")
parser.add_argument("--lr", type=float, default=1.25e-4, help="initial learning rate")
parser.add_argument("--log_dir", type=str, default="vocsave", help="default savedir")
parser.add_argument("--load_weights", type=str, default="", help="path to load weights of a model to continue training")
parser.add_argument("--initial_epoch", type=int, default=0, help="what is initial model")
parser.add_argument("--eval_freq", type=int, default=5, help="how often to evaluate (epoch)")
parser.add_argument("--max_objects", type=int, default=50, help="max number of detected objects")
parser.add_argument("--map_score_threshold", type=float, default=0.3, help="score threshold for mean average precision")
parser.add_argument("--map_iou_threshold", type=float, default=0.5, help="iou threshold for mean average precision")
parser.add_argument("--max_shuffle", type=int, default=10000, help="train shuffle samples")
parser.add_argument("--num_parallel_calls", type=int, default=-1, help="parallel calls for mapping, -1 for autotune")
parser.add_argument("--prefetch", type=int, default=-1, help="how many batches to prefetch, -1 for autotune")
parser.add_argument(
    "--keep_aspect_ratio",
    dest="keep_aspect_ratio",
    action="store_true",
    help="False (default) if the image is stretched to NN input size.",
)
parser.add_argument(
    "--no_log_images",
    dest="no_log_images",
    action="store_true",
    help="If we should show inputs and results images in tensorboard",
)
args = parser.parse_args()

# load dataset
dataset = None
if args.dataset == "voc":
    dataset = VocDataset(args.lr)
elif args.dataset == "coco":
    dataset = CocoDataset(args.lr)
elif args.dataset == "custom":
    dataset = CustomDataset(args.dataset_path_tr, args.dataset_path_te, args.lr, args.dataset_prefix)
else:
    print(f"Unknown dataset {args.dataset}.")
    exit()

backbone = XModelBackbone[args.backbone.upper()]
model_type = XModelType[args.model_type.upper()]
mode = XModelMode[args.model_mode.upper()]

# prepare model configuration
model_config = ModelConfig(
    args.image_size,
    dataset.classes,
    args.max_objects,
    size_variation=args.size_variation,
    keep_aspect_ratio=args.keep_aspect_ratio,
    model_type=model_type,
)

# augmentation config
hard_augmentation = HardAugmentation(0.7)
easy_augmentation = EasyAugmentation(0.3)

scheduler_cb = tf.keras.callbacks.LearningRateScheduler(dataset.scheduler)
# optimizer = tf.keras.optimizers.SGD(dataset.scheduler(args.initial_epoch), momentum=0.9)
optimizer = tf.keras.optimizers.Adam(dataset.scheduler(args.initial_epoch))

train_processing = BatchPreprocessing(model_config, train=True, augmentations=[hard_augmentation, easy_augmentation])
train_dataset, train_examples = dataset.load_train_datasets()
ds = (
    train_dataset.shuffle(min(args.max_shuffle, train_examples), reshuffle_each_iteration=True)
    .map(dataset.decode, num_parallel_calls=args.num_parallel_calls)
    .map(train_processing.prepare_for_batch, num_parallel_calls=args.num_parallel_calls)
    .batch(args.batch_size)
    .map(train_processing.preprocess_batch, num_parallel_calls=args.num_parallel_calls)
    .prefetch(args.prefetch)
)

# validation dataset
validation_processing = BatchPreprocessing(model_config, train=False)
validation_dataset, validation_examples = dataset.load_validation_datasets()
dataset_validation = (
    validation_dataset.map(dataset.decode, num_parallel_calls=args.num_parallel_calls)
    .map(validation_processing.prepare_for_batch, num_parallel_calls=args.num_parallel_calls)
    .batch(args.batch_size)
    .map(validation_processing.preprocess_batch, num_parallel_calls=args.num_parallel_calls)
    .prefetch(args.prefetch)
)

strategy = tf.distribute.MirroredStrategy()
print("Number of gpu devices: {}".format(strategy.num_replicas_in_sync))

with strategy.scope():
    if args.pretrained:
        print("Loading a pretrained model, creating new output layers.")
        model = load_and_update_model(args.pretrained, model_config.labels, model_type)
    else:
        print("Creating a new model.")
        model = create_model(
            None,
            model_config.labels,
            backbone=backbone,
            mode=mode,
            pretrained_backbone=not args.random_weights,
            model_type=model_config.model_type,
        )

        if args.pretrained_weights:
            # when finetuning right now only with .h5 format, there is some bug when loading from saved_model format
            load_pretrained_weights(model, args.pretrained_weights)

    if args.load_weights:
        model.load_weights(args.load_weights)

    model.compile(optimizer=optimizer, loss=model.get_loss_funcs())
    model.summary()

# we need to save right now just weights
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(args.log_dir, "checkpoints", "model_{epoch}"),
    save_freq="epoch",
    verbose=1,
    save_best_only=False,
    save_weights_only=True,
)
model_checkpoint.set_model(model)

log_dir = os.path.join(args.log_dir, "logs")
image_log = ImageLog(ds, model_config, log_dir=log_dir)
result_log = ResultImageLogCallback(dataset_validation, model_config, model, freq=args.eval_freq, log_dir=log_dir)
tensorboard = XTensorBoardCallback(log_dir=log_dir, update_freq="epoch", histogram_freq=args.eval_freq)
mapCallback = MAPValidationCallback(
    log_dir,
    dataset_validation,
    model,
    model_config.max_objects,
    model_config.labels,
    args.map_iou_threshold,
    args.map_score_threshold,
)

callbacks = [scheduler_cb, tensorboard, model_checkpoint, mapCallback]
if not args.no_log_images:
    callbacks += [image_log, result_log]

model.fit(
    ds,
    epochs=args.epochs,
    initial_epoch=args.initial_epoch,
    validation_data=dataset_validation,
    validation_freq=args.eval_freq,
    callbacks=callbacks,
)

model.save_weights(os.path.join(args.log_dir, "checkpoints", "final_weights"))
model.save_weights(os.path.join(args.log_dir, "model.h5"))
