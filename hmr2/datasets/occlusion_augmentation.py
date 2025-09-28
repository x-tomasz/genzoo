import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import cv2
import PIL.Image

_ANIMAL = {"bird", "cat", "cow", "dog", "horse", "sheep"}

def load_occluders(root: str, min_px: int = 500, scale: float = 0.8):
    """
    Load occluder objects from VOC dataset.
    
    Args:
        root (str): Path to VOC dataset root directory
        min_px (int): Minimum pixel area for occluders
        scale (float): Scale factor for occluders
        
    Returns:
        list: List of occluder images with alpha channels
    """
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    png = lambda fn: fn.replace("jpg", "png")
    oc = []
    
    annotations_dir = f"{root}/Annotations"
    jpeg_dir = f"{root}/JPEGImages"
    segmentation_dir = f"{root}/SegmentationObject"
    
    if not os.path.exists(annotations_dir):
        return oc
    
    annotation_files = sorted(os.listdir(annotations_dir))
    
    for ann in annotation_files:
        try:
            xml = ET.parse(f"{annotations_dir}/{ann}").getroot()
            if xml.find("segmented").text != "1":
                continue
                
            img = np.asarray(PIL.Image.open(f"{jpeg_dir}/{xml.find('filename').text}"))
            lbl = np.asarray(PIL.Image.open(f"{segmentation_dir}/{png(xml.find('filename').text)}"))
            
            for i, o in enumerate(xml.findall("object")):
                if o.find("name").text in _ANIMAL or o.find("difficult").text != "0" or o.find("truncated").text != "0":
                    continue
                    
                b = o.find("bndbox")
                x1, y1, x2, y2 = [int(b.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax")]
                m = (lbl[y1:y2, x1:x2] == i + 1).astype(np.uint8) * 255
                
                if cv2.countNonZero(m) < min_px:
                    continue
                    
                m[cv2.erode(m, se) < m] = 192
                rgba = np.concatenate([img[y1:y2, x1:x2], m[..., None]], -1)
                oc.append(cv2.resize(rgba, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
                
        except Exception as e:
            continue
    
    return oc

def apply_occlusion_augmentation(img_patch: np.ndarray, occluders: list, fraction: tuple = (0.5, 0.8), probability: float = 0.3) -> np.ndarray:
    """
    Apply occlusion augmentation to an image patch.
    
    Args:
        img_patch (np.ndarray): Input image patch (C, H, W)
        occluders (list): List of occluder images
        fraction (tuple): Range for occluder size fraction
        probability (float): Probability of applying occlusion
        
    Returns:
        np.ndarray: Augmented image patch
    """
    if not occluders or random.random() > probability:
        return img_patch
    
    # Convert from (C, H, W) to (H, W, C) for processing
    img_hwc = img_patch.transpose(1, 2, 0)
    dst = img_hwc.copy()
    
    h, w = dst.shape[:2]
    sticker = random.choice(occluders)
    sw, sh = sticker.shape[1], sticker.shape[0]
    
    f = random.uniform(*fraction)
    s = min(f * w / sw, f * h / sh)
    sticker = cv2.resize(sticker, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
    sw, sh = sticker.shape[1], sticker.shape[0]
    
    cx, cy = random.randrange(w), random.randrange(h)
    x0, y0 = max(cx - sw // 2, 0), max(cy - sh // 2, 0)
    x1, y1 = min(x0 + sw, w), min(y0 + sh, h)
    xs, ys = x0 - (cx - sw // 2), y0 - (cy - sh // 2)
    
    roi_dst = dst[y0:y1, x0:x1]
    roi_src = sticker[ys:ys + (y1 - y0), xs:xs + (x1 - x0)]
    
    # Ensure both regions have the same shape by cropping to the smaller dimensions
    actual_height, actual_width = roi_dst.shape[:2]
    src_height, src_width = roi_src.shape[:2]
    crop_height = min(actual_height, src_height)
    crop_width = min(actual_width, src_width)
    
    roi_dst = roi_dst[:crop_height, :crop_width]
    roi_src = roi_src[:crop_height, :crop_width]
    
    # Handle RGBA blending (both source and destination have 4 channels)
    if roi_src.shape[2] == 4 and roi_dst.shape[2] == 4:
        # Extract RGB and alpha channels from occluder
        rgb_src = roi_src[..., :3]
        alpha = roi_src[..., 3:4] / 255.0
        
        # Extract RGB from destination and apply alpha blending
        rgb_dst = roi_dst[..., :3]
        blended_rgb = (alpha * rgb_src + (1 - alpha) * rgb_dst).astype(np.uint8)
        
        # Keep original alpha from destination
        alpha_dst = roi_dst[..., 3:4]
        
        # Combine blended RGB with destination alpha
        blended = np.concatenate([blended_rgb, alpha_dst], axis=2)
        dst[y0:y0+crop_height, x0:x0+crop_width] = blended
    
    # Convert back to (C, H, W)
    result = dst.transpose(2, 0, 1)
    return result

# Global variable to store loaded occluders
_loaded_occluders = None

def get_occluders(voc_root: str = "/tmp/VOC", min_px: int = 500, scale: float = 0.8):
    """
    Get or load occluders from VOC dataset.
    
    Args:
        voc_root (str): Path to VOC dataset
        min_px (int): Minimum pixel area for occluders
        scale (float): Scale factor for occluders
        
    Returns:
        list: List of occluder images
    """
    global _loaded_occluders
    
    if _loaded_occluders is None:
        _loaded_occluders = load_occluders(voc_root, min_px, scale)
    
    return _loaded_occluders 