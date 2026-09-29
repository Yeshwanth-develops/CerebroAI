"""
Explainable AI (XAI) Module: Grad-CAM for Brain MRI Classification.
Generates attention heatmaps showing anatomical regions contributing to AD diagnosis.
"""

from typing import Optional, Tuple
import cv2
import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
from tensorflow import keras


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: keras.Model,
    last_conv_layer_name: str = "bn5b_branch2b",
    pred_index: Optional[int] = None,
) -> np.ndarray:
    """
    Computes a Grad-CAM heatmap for a given input MRI image and target convolution layer.

    Args:
        img_array: Shape (1, H, W, C) or (H, W, C), float normalized [0, 1].
        model: Trained Keras ResNet-18 model.
        last_conv_layer_name: Name of the convolutional layer to inspect.
        pred_index: Class index (0 for Normal, 1 for AD, or None for top predicted class).

    Returns:
        2D numpy array heatmap of shape (H, W) normalized to [0, 1].
    """
    if len(img_array.shape) == 3:
        img_array = np.expand_dims(img_array, axis=0)

    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        conv_layers = [l.name for l in model.layers if "res5" in l.name or "conv" in l.name]
        last_conv_layer_name = conv_layers[-1] if conv_layers else model.layers[-4].name
        last_conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = keras.Model(
        inputs=[model.inputs],
        outputs=[last_conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            if predictions.shape[-1] == 1:
                class_channel = predictions[0]
            else:
                pred_index = tf.argmax(predictions[0])
                class_channel = predictions[:, pred_index]
        else:
            if predictions.shape[-1] == 1:
                class_channel = predictions[0]
            else:
                class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0.0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def overlay_gradcam(
    img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Superimposes the Grad-CAM heatmap onto the original grayscale MRI slice.

    Args:
        img: Original slice of shape (H, W) or (H, W, 1), normalized [0, 1] or uint8 [0, 255].
        heatmap: 2D array of shape (H, W) normalized [0, 1].
        alpha: Transparency weighting for the heatmap overlay.
        colormap: OpenCV colormap (default COLORMAP_JET).

    Returns:
        RGB superimposed image array of shape (H, W, 3) in uint8 [0, 255].
    """
    if img.max() <= 1.0:
        base_img = np.uint8(255 * img.squeeze())
    else:
        base_img = np.uint8(img.squeeze())

    base_rgb = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB) if len(base_img.shape) == 2 else base_img

    heatmap_resized = cv2.resize(heatmap, (base_rgb.shape[1], base_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    superimposed = cv2.addWeighted(colored_heatmap, alpha, base_rgb, 1.0 - alpha, 0)
    return superimposed
