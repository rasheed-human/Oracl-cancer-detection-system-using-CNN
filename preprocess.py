import cv2
import numpy as np

TARGET_SIZE = (224, 224)

def _apply_clahe(img_rgb):
    """Applies CLAHE to the L-channel of an RGB image."""
    # Convert RGB to LAB
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    
    # Split LAB channels
    l_channel, a_channel, b_channel = cv2.split(img_lab)
    
    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    # Merge the CLAHE-enhanced L-channel back with A and B channels
    merged_channels = cv2.merge([cl, a_channel, b_channel])
    
    # Convert LAB back to RGB
    final_img_rgb = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2RGB)
    
    return final_img_rgb

def preprocess_image_from_path(img_path):
    """Loads and preprocesses an image from a file path."""
    # Read image in BGR format
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found at path: {img_path}")
        
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Call the common processing function
    return _process_image(img_rgb)

def preprocess_image_from_array(img_array):
    """Preprocesses an image from a numpy array (assumed to be RGB)."""
    # Call the common processing function
    return _process_image(img_array)

def _process_image(img_rgb):
    """Common image processing pipeline."""
    # Apply CLAHE for contrast enhancement
    img_clahe = _apply_clahe(img_rgb)
    
    # Resize image
    img_resized = cv2.resize(img_clahe, TARGET_SIZE)
    
    # Normalize pixel values to [0, 1]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    return img_normalized