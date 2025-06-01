import numpy as np
from PIL import Image

def estimate_capacity(img, channels_per_pixel=3, pixels_used_per_block=3):
    h, w, _ = img.shape
    blocks = (h // 2) * (w // 2)
    return blocks * pixels_used_per_block * channels_per_pixel


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

def encode_length_and_data(data_bits: str) -> str:
    length = len(data_bits)
    length_bits = format(length, '032b')
    return length_bits + data_bits

def decode_length_and_data(extracted_bits: str) -> str:
    if len(extracted_bits) < 32:
        return ''
    length = int(extracted_bits[:32], 2)
    return extracted_bits[32:32 + length]

