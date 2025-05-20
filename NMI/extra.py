import numpy as np
from PIL import Image

def estimate_capacity(img):
    w, h, c = img.shape
    return w * h * c

def text_to_bitstring(text):
    return ''.join(format(byte, '08b') for byte in text.encode('utf-8'))


def bitstring_to_text(bitstring):
    bytes_list = [int(bitstring[i:i+8], 2) for i in range(0, len(bitstring), 8)]
    return bytes(bytes_list).decode('utf-8', errors='ignore')

def safe_slice(b, start, end=None):

    if end is None:
        end = len(b)
    if start < 0:
        start = max(0, len(b) + start)
    if end < 0:
        end = max(0, len(b) + end)
    return b[start:end] if start < len(b) and start < end else ''

def extract_bits(data_bits, start, length):

    bits = data_bits[start:start + length]
    bits += '0' * (length - len(bits))
    return bits

def image_to_bitstring(img: np.ndarray) -> str:
    flat = img.flatten()
    return ''.join(format(byte, '08b') for byte in flat)

def bitstring_to_image(bitstring: str, shape: tuple) -> np.ndarray:
    
    bytes_array = [int(bitstring[i:i+8], 2) for i in range(0, len(bitstring), 8)]
    return np.array(bytes_array, dtype=np.uint8).reshape(shape)
