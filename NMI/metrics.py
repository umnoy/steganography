from extra import text_to_bitstring
import numpy as np
from skimage.metrics import structural_similarity as ssim
import math

def mse(original, modified):
    return np.mean((original.astype("float") - modified.astype("float")) ** 2)

def rmse(original, modified):
    return np.sqrt(mse(original, modified))

def psnr(original, modified):
    mse_val = mse(original, modified)
    if mse_val == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse_val))

def ssim_index(original, modified):
    if original.ndim == 3:
        ssim_total = 0
        for i in range(3):
            ssim_val = ssim(original[..., i], modified[..., i], data_range=255)
            ssim_total += ssim_val
        return ssim_total / 3
    else:
        return ssim(original, modified, data_range=255)


def ec(text, image):
    num_pixels = image.shape[0] * image.shape[1]
    bits = len(text_to_bitstring(text))

    return (bits / num_pixels)


def embedding_capacity(image_shape):
    h, w = image_shape[:2]
    return (h * 2 - 1) * (w * 2 - 1) - h * w 


def ber(original_bits, extracted_bits):
    length = min(len(original_bits), len(extracted_bits))
    errors = sum(o != e for o, e in zip(original_bits[:length], extracted_bits[:length]))
    return errors / length

def ncc(original, modified):
    original = original.astype(np.float64)
    modified = modified.astype(np.float64)
    numerator = np.sum((original - np.mean(original)) * (modified - np.mean(modified)))
    denominator = np.sqrt(np.sum((original - np.mean(original)) ** 2) * np.sum((modified - np.mean(modified)) ** 2))
    return numerator / denominator if denominator != 0 else 0

