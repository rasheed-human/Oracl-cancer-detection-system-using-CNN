import tensorflow as tf
import numpy as np
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap.
    
    Args:
        img_array: Preprocessed input image array (with batch dimension).
        model: The CNN model (or the CNN branch of the hybrid model).
        last_conv_layer_name: The name of the last convolutional layer.
        pred_index: The index of the class to visualize. (None for binary/regression).
    """
    
    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            # For binary classification or single-output regression
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization, we will normalize the heatmap between 0 & 1
    # We also apply ReLU to only see positive contributions
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_heatmap(original_img_bgr, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Applies a heatmap overlay to an original image.
    
    Args:
        original_img_bgr: The *original* image (before preprocessing) in BGR format (0-255).
                          Must be the same size as the heatmap (e.g., 224x224).
        heatmap: The 2D Grad-CAM heatmap (0-1).
        alpha: Transparency of the heatmap.
    """
    if original_img_bgr.shape[:2] != heatmap.shape:
        # Resize heatmap to match image dimensions
        heatmap = cv2.resize(heatmap, (original_img_bgr.shape[1], original_img_bgr.shape[0]))

    # Convert heatmap to 8-bit unsigned integer
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    
    # Apply the colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Add weighted overlay
    superimposed_img = cv2.addWeighted(original_img_bgr, 1 - alpha, heatmap_colored, alpha, 0)
    
    return superimposed_img