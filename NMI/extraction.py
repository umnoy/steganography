from PIL import Image
import numpy as np
import math


def extract_reference_image(embedded_img):

    h, w = embedded_img.shape[:2]
    if embedded_img.ndim == 3:
        channels = embedded_img.shape[2]
        ref_img = np.zeros((h // 2, w // 2, channels), dtype=np.uint8)
        for i in range(0, h, 2):
            for j in range(0, w, 2):
                ref_img[i // 2, j // 2] = embedded_img[i, j]
    else:
        ref_img = np.zeros((h // 2, w // 2), dtype=np.uint8)
        for i in range(0, h, 2):
            for j in range(0, w, 2):
                ref_img[i // 2, j // 2] = embedded_img[i, j]

    return ref_img


def extract_bits_from_difference(embedded_img, interp_img):
    h, w = embedded_img.shape[:2]
    extracted_bits = ''

    is_color = (embedded_img.ndim == 3)
    channels = embedded_img.shape[2] if is_color else 1

    for i in range(0, h, 2):
        for j in range(0, w, 2):
            for c in range(channels) if is_color else [None]:
                if is_color:
                    block_emb = embedded_img[i:i+2, j:j+2, c]
                    block_int = interp_img[i:i+2, j:j+2, c]
                else:
                    block_emb = embedded_img[i:i+2, j:j+2]
                    block_int = interp_img[i:i+2, j:j+2]

                d1 = int(block_int[0,1]) - int(block_int[0,0])
                d2 = int(block_int[1,0]) - int(block_int[0,0])
                d3 = int(block_int[1,1]) - int(block_int[0,0])

                n1 = int(math.log2(abs(d1))) if abs(d1) > 1 else 0
                n2 = int(math.log2(abs(d2))) if abs(d2) > 1 else 0
                n3 = int(math.log2(abs(d3))) if abs(d3) > 1 else 0

                if n1 > 0:
                    m1 = (int(block_emb[0,1]) - int(block_int[0,1])) & ((1 << n1) - 1)
                    extracted_bits += format(m1, f'0{n1}b')
                if n2 > 0:
                    m2 = (int(block_emb[1,0]) - int(block_int[1,0])) & ((1 << n2) - 1)
                    extracted_bits += format(m2, f'0{n2}b')
                if n3 > 0:
                    m3 = (int(block_emb[1,1]) - int(block_int[1,1])) & ((1 << n3) - 1)
                    extracted_bits += format(m3, f'0{n3}b')

    return extracted_bits
