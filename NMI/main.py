from PIL import Image
import numpy as np
import math

from interpolation import *
from extra import *
from embed import *
from extraction import *
from metrics import *

def calculate_metrics(original_img, stego_img):
    mse = np.mean((original_img - stego_img) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return mse, psnr


while True:
    try:
       img = Image.open(input("Укажите путь к изображению-контейнеру: "))
       break
    except Exception:
            print("Укажите верный путь!")


while True:
    try:
        mode = int(input("Выберите режим (1 - Встраивание, 2 - Извлечение)\n\
--> "))
        if mode != 1 and mode != 2:
             raise ValueError
        break
    except Exception:
            print("Выберите 1 или 2!")

option = 1

if mode == 1:
     while True:
        try:
            if option == 1:
                text = input("Введите текст для встраивания: ")
                if text:
                    break
                else:
                    raise ValueError
            if option == 2:
                secret = Image.open(input("Укажите путь до изображения: "))
                if secret:
                    break
                else:
                    raise Exception
        except ValueError:
                print("Текст не введен")
        except Exception:
             print("Введите путь еще раз")


arr = np.array(img)


if mode == 1:

    bits = encode_length_and_data(text_to_bitstring(100 * text))
    interpolated = nmi_interpolation(arr)
    img_int = Image.fromarray(interpolated)
    img_int.save("interpolated.png")

    embedded = embed(interpolated, bits)
    emb = Image.fromarray(embedded)
    emb.save("embedded.png")
    print("\n")
    print("MSE:", mse(interpolated, embedded))
    print("RMSE:", rmse(interpolated, embedded))
    print("PSNR:", psnr(interpolated, embedded))
    print("SSIM:", ssim_index(interpolated, embedded))
    print("Результат в embedded.png и interpolated.png")




if mode == 2:
     
     ref = extract_reference_image(arr)
     raw_bits = extract_bits_from_difference(arr, nmi_interpolation(ref))
     clean_bits = decode_length_and_data(raw_bits)
     message = bitstring_to_text(clean_bits)

     print("Извлеченное сообщение: ", message)
     print("Длина извлеченного сообщения: ", len(str(message)))
     print("EC: ", ec(clean_bits, arr))

