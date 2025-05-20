import numpy as np
from PIL import Image

def estimate_capacity_rgb(image_array):
    if len(image_array.shape) != 3 or image_array.shape[2] != 3:
        raise ValueError("Ожидается RGB-изображение с 3 каналами.")

    h, w, _ = image_array.shape

    h_interp = h * 2 - 1
    w_interp = w * 2 - 1

    total_pixels = h_interp * w_interp
    reference_pixels = h * w
    embeddable_pixels_per_channel = total_pixels - reference_pixels

    total_embeddable_bits = embeddable_pixels_per_channel * 3

    return {
        "bits": total_embeddable_bits,
        "bytes": total_embeddable_bits // 8
    }




img = Image.open("./img/2.png").convert("RGB")
arr = np.array(img)

capacity = estimate_capacity_rgb(arr)

print(f"Максимальная вместимость: {capacity['bits']} битов ({capacity['bytes']} байт)")
