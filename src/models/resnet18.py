"""
Custom ResNet-18 Architecture for T1-Weighted Brain MRI Classification.
Built with TensorFlow 2.x / Keras with LeakyReLU activations and custom strides.
"""

from typing import List, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _identity_block(
    input_tensor: tf.Tensor,
    kernel_size: int,
    filters: List[int],
    stage: int,
    block: str,
) -> tf.Tensor:
    """Residual Identity Block with no downsampling."""
    filters1, filters2 = filters
    conv_base = f"res{stage}{block}_branch"
    bn_base = f"bn{stage}{block}_branch"

    x = layers.Conv2D(
        filters1,
        kernel_size,
        padding="same",
        kernel_initializer="he_normal",
        name=conv_base + "2a",
    )(input_tensor)
    x = layers.BatchNormalization(name=bn_base + "2a")(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)

    x = layers.Conv2D(
        filters2,
        kernel_size,
        padding="same",
        kernel_initializer="he_normal",
        name=conv_base + "2b",
    )(x)
    x = layers.BatchNormalization(name=bn_base + "2b")(x)

    x = layers.add([x, input_tensor])
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    return x


def _convolutional_block(
    input_tensor: tf.Tensor,
    kernel_size: int,
    filters: List[int],
    stage: int,
    block: str,
    strides: Tuple[int, int] = (2, 2),
) -> tf.Tensor:
    """Residual Convolutional Block with strided downsampling."""
    filters1, filters2 = filters
    conv_base = f"res{stage}{block}_branch"
    bn_base = f"bn{stage}{block}_branch"

    x = layers.Conv2D(
        filters1,
        kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer="he_normal",
        name=conv_base + "2a",
    )(input_tensor)
    x = layers.BatchNormalization(name=bn_base + "2a")(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)

    x = layers.Conv2D(
        filters2,
        kernel_size,
        padding="same",
        kernel_initializer="he_normal",
        name=conv_base + "2b",
    )(x)
    x = layers.BatchNormalization(name=bn_base + "2b")(x)

    shortcut = layers.Conv2D(
        filters2,
        (1, 1),
        strides=strides,
        kernel_initializer="he_normal",
        name=conv_base + "1",
    )(input_tensor)
    shortcut = layers.BatchNormalization(name=bn_base + "1")(shortcut)

    x = layers.add([x, shortcut])
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    return x


def build_resnet18(
    input_shape: Tuple[int, int, int] = (192, 160, 1),
    num_classes: int = 2,
    dropout_rate: float = 0.25,
    dense_units: int = 128,
) -> keras.Model:
    """
    Constructs a customized ResNet-18 model tailored for 2D Brain MRI slices.
    Includes final feature map hook for Grad-CAM explainability.
    """
    img_input = layers.Input(shape=input_shape, name="mri_input")

    x = layers.ZeroPadding2D(padding=(3, 3), name="conv1_pad")(img_input)
    x = layers.Conv2D(
        64, (7, 7), strides=(2, 2), padding="valid", kernel_initializer="he_normal", name="conv1"
    )(x)
    x = layers.BatchNormalization(name="bn_conv1")(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name="pool1")(x)

    x = _convolutional_block(x, 3, [64, 64], stage=2, block="a", strides=(1, 1))
    x = _identity_block(x, 3, [64, 64], stage=2, block="b")

    x = _convolutional_block(x, 3, [128, 128], stage=3, block="a", strides=(2, 2))
    x = _identity_block(x, 3, [128, 128], stage=3, block="b")

    x = _convolutional_block(x, 3, [256, 256], stage=4, block="a", strides=(2, 2))
    x = _identity_block(x, 3, [256, 256], stage=4, block="b")

    x = _convolutional_block(x, 3, [512, 512], stage=5, block="a", strides=(2, 2))
    x = _identity_block(x, 3, [512, 512], stage=5, block="b")

    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="head_dropout")(x)

    if dense_units > 0:
        x = layers.Dense(dense_units, kernel_initializer="he_normal", name="fc_features")(x)
        x = layers.LeakyReLU(negative_slope=0.1)(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)

    activation = "sigmoid" if num_classes == 1 or num_classes == 2 else "softmax"
    units = 1 if num_classes == 2 else num_classes
    outputs = layers.Dense(units, activation=activation, name="diagnosis_output")(x)

    model = keras.Model(inputs=img_input, outputs=outputs, name="ADNI_ResNet18")
    return model
