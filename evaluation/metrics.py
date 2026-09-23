import os
import sys
import argparse
import json
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import CLIPProcessor, CLIPModel
from concurrent.futures import ThreadPoolExecutor, as_completed


GLOBAL_CLIP_MODEL = None
GLOBAL_CLIP_PROCESSOR = None
GLOBAL_LPIPS_MODEL = None

def ensure_clip_loaded():
    """
    Lazily load the global CLIP model and processor once per process.
    """
    global GLOBAL_CLIP_MODEL, GLOBAL_CLIP_PROCESSOR
    if GLOBAL_CLIP_MODEL is None or GLOBAL_CLIP_PROCESSOR is None:
        GLOBAL_CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        GLOBAL_CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        if torch.cuda.is_available():
            GLOBAL_CLIP_MODEL = GLOBAL_CLIP_MODEL.to("cuda")
            GLOBAL_CLIP_MODEL.eval()

def ensure_lpips_loaded():
    """
    Lazily load the global LPIPS model once per process.

    Uses torchmetrics' AlexNet-backbone LPIPS with ``normalize=True`` so it
    accepts images in [0, 1] (it internally maps to [-1, 1]).
    """
    global GLOBAL_LPIPS_MODEL
    if GLOBAL_LPIPS_MODEL is None:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        GLOBAL_LPIPS_MODEL = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        )
        GLOBAL_LPIPS_MODEL.eval()
        if torch.cuda.is_available():
            GLOBAL_LPIPS_MODEL = GLOBAL_LPIPS_MODEL.to("cuda")

def clip_similarity(image1, image2):
    """
    Compute the CLIP similarity between two PIL images.

    Args:
    image1 (PIL.Image): The first input image.
    image2 (PIL.Image): The second input image.

    Returns:
    float: The CLIP similarity between the two images.
    """
    if image1.size != image2.size:
        image2 = image2.resize(image1.size)

    # Ensure global model is initialized
    ensure_clip_loaded()

    # Preprocess the images
    images = [image1, image2]
    inputs = GLOBAL_CLIP_PROCESSOR(images=images, return_tensors="pt")
    device = next(GLOBAL_CLIP_MODEL.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Compute the features for the images
    with torch.no_grad():
        features = GLOBAL_CLIP_MODEL.get_image_features(**inputs)

    # Compute the cosine similarity between the image features
    sim = torch.nn.functional.cosine_similarity(features[0], features[1], dim=-1)

    return sim.item()

def photometric_loss(image1: Image.Image, image2: Image.Image) -> float:
    """
    Compute the photometric loss between two PIL images.

    Args:
    image1 (PIL.Image): The first input image.
    image2 (PIL.Image): The second input image.

    Returns:
    float: The photometric loss between the two images.
    """
    if image1.size != image2.size:
        image2 = image2.resize(image1.size)
    
    # Convert images to numpy arrays
    img1_array = np.array(image1)[:, :, :3]
    img2_array = np.array(image2)[:, :, :3]

    # Normalize images to [0, 1]
    img1_norm = img1_array.astype(np.float32) / 255.0
    img2_norm = img2_array.astype(np.float32) / 255.0

    # Compute the squared difference between the normalized images
    diff = np.square(img1_norm - img2_norm)

    # Compute the mean squared error
    mse = np.mean(diff)
    return mse

def lpips_distance(image1: Image.Image, image2: Image.Image) -> float:
    """
    Compute the LPIPS perceptual distance between two PIL images.

    Lower is better (0 = identical). Uses AlexNet backbone. Images are
    resized to a common size (image1's), converted to [0, 1] float tensors
    of shape (1, 3, H, W).

    Args:
    image1 (PIL.Image): The first input image.
    image2 (PIL.Image): The second input image.

    Returns:
    float: The LPIPS distance between the two images.
    """
    if image1.size != image2.size:
        image2 = image2.resize(image1.size)

    ensure_lpips_loaded()
    device = next(GLOBAL_LPIPS_MODEL.parameters()).device

    def to_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.array(img)[:, :, :3].astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    t1 = to_tensor(image1).to(device)
    t2 = to_tensor(image2).to(device)
    with torch.no_grad():
        d = GLOBAL_LPIPS_MODEL(t1, t2)
    return float(d)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    img1 = Image.fromarray(rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8))
    img2 = Image.fromarray(rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8))
    print("clip_similarity:", clip_similarity(img1, img2))
    print("photometric_loss:", photometric_loss(img1, img2))
    print("lpips_distance:", lpips_distance(img1, img2))