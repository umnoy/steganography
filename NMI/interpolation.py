import numpy as np
from PIL import Image

import numpy as np

def interpolate_block(block_2x2):
    if block_2x2.shape != (2, 2):
        return np.zeros((4, 4), dtype=np.uint8)
    
    top_left     = int(block_2x2[0, 0])
    top_right    = int(block_2x2[0, 1])
    bottom_left  = int(block_2x2[1, 0])
    bottom_right = int(block_2x2[1, 1])

    interp_block = np.zeros((4, 4), dtype=np.uint8)

    interp_block[0, 0] = top_left
    interp_block[0, 1] = (top_left + top_right) // 2
    interp_block[0, 2] = top_right
    interp_block[1, 0] = (top_left + bottom_left) // 2
    interp_block[1, 2] = (top_right + bottom_right) // 2
    interp_block[1, 1] = (top_left + int(interp_block[0, 1]) + int(interp_block[1, 0])) // 3
    interp_block[2, 0] = bottom_left
    interp_block[2, 1] = (bottom_left + bottom_right) // 2
    interp_block[2, 2] = bottom_right

    interp_block[0, 3] = (top_right + int(interp_block[0, 1])) // 2
    interp_block[1, 3] = (int(interp_block[1, 2]) + int(interp_block[1, 1])) // 2
    interp_block[2, 3] = (int(interp_block[2, 2]) + int(interp_block[2, 1])) // 2

    interp_block[3, 0] = (int(interp_block[2, 0]) + int(interp_block[1, 0])) // 2
    interp_block[3, 1] = (int(interp_block[2, 1]) + int(interp_block[1, 1])) // 2
    interp_block[3, 2] = (int(interp_block[2, 2]) + int(interp_block[1, 2])) // 2
    interp_block[3, 3] = (int(interp_block[3, 2]) + int(interp_block[2, 2]) + int(interp_block[2, 3])) // 3

    return interp_block.astype(np.uint8)


def nmi_interpolation(img_array):
    h, w = img_array.shape[:2]
    img_array = img_array[:h - h % 2, :w - w % 2]
    
    if img_array.ndim == 3:
        is_color = True
        channels = img_array.shape[2]
    else:
        is_color = False
        channels = 1

    new_h, new_w = (h // 2) * 4, (w // 2) * 4
    result = np.zeros((new_h, new_w, channels), dtype=np.uint8) if is_color else np.zeros((new_h, new_w), dtype=np.uint8)

    for i in range(0, h - 1, 2):
        for j in range(0, w - 1, 2):
            ni = (i // 2) * 4
            nj = (j // 2) * 4

            if is_color:
                for c in range(channels):
                    block_2x2_channel = img_array[i:i+2, j:j+2, c]
                    interp_channel = interpolate_block(block_2x2_channel)
                    result[ni:ni+4, nj:nj+4, c] = interp_channel
            else:
                block_2x2 = img_array[i:i+2, j:j+2]
                interp = interpolate_block(block_2x2)
                result[ni:ni+4, nj:nj+4] = interp

    return result
