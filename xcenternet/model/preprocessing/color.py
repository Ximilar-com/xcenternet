import numpy as np
import tensorflow as tf
from imgaug import augmenters as iaa

sometimes_010 = lambda aug: iaa.Sometimes(0.10, aug)

blurs = iaa.Sequential(
    [
        sometimes_010(
            iaa.OneOf(
                [
                    iaa.GaussianBlur(sigma=(0, 1.5)),
                    iaa.AverageBlur(k=(1, 5)),
                    iaa.MedianBlur(k=(1, 5)),
                    iaa.MotionBlur(k=(3, 5)),
                ]
            )
        )
    ]
)

contrasts = iaa.Sequential(
    [
        sometimes_010(
            iaa.OneOf(
                [
                    iaa.LogContrast((0.8, 1.2)),
                    iaa.GammaContrast((0.8, 1.2)),
                    iaa.LinearContrast((0.8, 1.2)),
                    iaa.Alpha((0.0, 1.0), iaa.AllChannelsHistogramEqualization()),
                    iaa.Alpha((0.0, 1.0), iaa.HistogramEqualization()),
                    iaa.CLAHE(clip_limit=(1, 3)),
                    iaa.AllChannelsCLAHE(clip_limit=(1, 3)),
                ]
            )
        )
    ]
)

dropouts = iaa.Sequential(
    [
        sometimes_010(
            iaa.OneOf(
                [
                    iaa.Dropout(p=0.01, per_channel=True),
                    iaa.Dropout(p=0.01, per_channel=False),
                    iaa.Cutout(fill_mode="constant", cval=(0, 255), size=(0.1, 0.4), fill_per_channel=0.5),
                    iaa.CoarseDropout((0.0, 0.08), size_percent=(0.02, 0.25), per_channel=0.5),
                    iaa.SaltAndPepper(p=0.01, per_channel=True),
                    iaa.SaltAndPepper(p=0.01, per_channel=False),
                    iaa.AdditiveLaplaceNoise(scale=0.02 * 255, per_channel=True),
                    iaa.AdditiveLaplaceNoise(scale=0.02 * 255, per_channel=False),
                    iaa.AdditiveGaussianNoise(scale=0.02 * 255, per_channel=True),
                    iaa.AdditiveGaussianNoise(scale=0.02 * 255, per_channel=False),
                    iaa.AdditivePoissonNoise(lam=4.0, per_channel=True),
                    iaa.AdditivePoissonNoise(lam=4.0, per_channel=False),
                ]
            )
        )
    ]
)


@tf.function
def tf_py_blur(image):
    im_shape = image.shape
    [image,] = tf.py_function(imgaug_blur, [image], [tf.uint8])
    image.set_shape(im_shape)
    return image


@tf.function
def tf_py_contrast(image):
    im_shape = image.shape
    [image,] = tf.py_function(imgaug_contrast, [image], [tf.uint8])
    image.set_shape(im_shape)
    return image


@tf.function
def tf_py_dropout(image):
    im_shape = image.shape
    [image,] = tf.py_function(imgaug_dropout, [image], [tf.uint8])
    image.set_shape(im_shape)
    return image


def imgaug_blur(image):
    return blurs.augment_images([image.numpy().astype(np.uint8)])[0]


def imgaug_dropout(image):
    return dropouts.augment_images([image.numpy().astype(np.uint8)])[0]


def imgaug_contrast(image):
    return contrasts.augment_images([image.numpy().astype(np.uint8)])[0]
