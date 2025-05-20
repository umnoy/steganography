from PIL import Image
import numpy as np
import math

def embed(img_array, data_bits):

    if img_array.ndim == 3:
        h, w, channels = img_array.shape
    elif img_array.ndim == 2:
        h, w = img_array.shape
        channels = 1
        img_array = img_array[:, :, np.newaxis]
    else:
        raise ValueError("Unsupported image format")

    h -= h % 2
    w -= w % 2
    img = img_array[:h, :w].copy()
    embedded = img.copy()

    bit_str = ''.join(data_bits)
    bit_index = 0
    total_bits = len(bit_str)

    for i in range(0, h, 2):
        for j in range(0, w, 2):
            for c in range(channels):
                if bit_index >= total_bits:
                    break

                block = img[i:i+2, j:j+2, c]
                top_left = int(block[0, 0])

                d1 = int(block[0, 1]) - top_left
                d2 = int(block[1, 0]) - top_left
                d3 = int(block[1, 1]) - top_left

                n1 = int(math.log2(abs(d1))) if abs(d1) > 1 else 0
                n2 = int(math.log2(abs(d2))) if abs(d2) > 1 else 0
                n3 = int(math.log2(abs(d3))) if abs(d3) > 1 else 0


                if n1 > 0 and bit_index < total_bits:
                    m1 = int(bit_str[bit_index:bit_index+n1].ljust(n1, '0'), 2)
                    block[0, 1] = np.clip(int(block[0, 1]) + m1, 0, 255)
                    bit_index += n1

                if n2 > 0 and bit_index < total_bits:
                    m2 = int(bit_str[bit_index:bit_index+n2].ljust(n2, '0'), 2)
                    block[1, 0] = np.clip(int(block[1, 0]) + m2, 0, 255)
                    bit_index += n2

                if n3 > 0 and bit_index < total_bits:
                    m3 = int(bit_str[bit_index:bit_index+n3].ljust(n3, '0'), 2)
                    block[1, 1] = np.clip(int(block[1, 1]) + m3, 0, 255)
                    bit_index += n3

                embedded[i:i+2, j:j+2, c] = block

            if bit_index >= total_bits:
                return embedded.squeeze()

    return embedded.squeeze()
