import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from embed import embed
from extraction import extract_bits_from_difference, extract_reference_image
from interpolation import nmi_interpolation
from extra import text_to_bitstring, bitstring_to_text, estimate_capacity
from skimage.metrics import peak_signal_noise_ratio as psnr

def load_image(path):
    return np.array(Image.open(path).convert("RGB"))


def save_image(array, path):
    Image.fromarray(array).save(path)


def test_max_capacity(img_arr):
    print("Тест 1: Максимальный объем встраивания")
    max_bits = estimate_capacity(img_arr)
    message = "a" * (max_bits // 8)
    interpolated = nmi_interpolation(img_arr)
    stego = embed(interpolated, text_to_bitstring(message))
    quality = psnr(interpolated, stego)
    extracted = extract_bits_from_difference(stego, interpolated)
    extracted = bitstring_to_text(extracted)
    print(f"Встроено: {len(extracted)} символов")
    print(f"PSNR: {quality:.2f} dB\n")


def test_different_payloads(img_arr):
    print("Тест 2: Разные объемы встраивания")
    for percent in [25, 50, 75, 100]:
        bits = estimate_capacity(img_arr) * percent // 100
        message = "a" * (bits // 8)
        interpolated = nmi_interpolation(img_arr)
        stego = embed(interpolated, text_to_bitstring(message))
        quality = psnr(interpolated, stego)
        print(f"Загрузка: {percent}% | PSNR: {quality:.2f} dB")
    print()


def test_robustness(img_arr):
    print("Тест 3: Робастность")
    message = 1000 * "robustness_test"
    interpolated = nmi_interpolation(img_arr)
    stego = embed(interpolated, text_to_bitstring(message))
    stego_img = Image.fromarray(stego)

    ref = extract_reference_image(stego)
    extracted = extract_bits_from_difference(stego, nmi_interpolation(ref))
    extracted_text = bitstring_to_text(extracted)
    correct_ratio = sum(a == b for a, b in zip(extracted_text, message)) / len(message) * 100
    print(f"Было встроено 1000 раз сообщение: {message[:15]}")
    print(f"Точность извлечения без сжатия: {correct_ratio:.2f}%")

    blurred = np.array(stego_img.filter(ImageFilter.GaussianBlur(radius=1)))
    ref_blur = extract_reference_image(blurred)
    extracted_blur = extract_bits_from_difference(blurred, nmi_interpolation(ref_blur))
    blur_text = bitstring_to_text(extracted_blur)
    correct_ratio_blur = sum(a == b for a, b in zip(blur_text, message)) / len(message) * 100
    #print(f"После блюра извлечено: {blur_text}")
    print(f"Точность извлечения после размытия: {correct_ratio_blur:.2f}%")

    rgb_stego_img = stego_img.convert("RGB")
    rgb_stego_img.save("temp.jpg", quality=85)
    jpeg = np.array(Image.open("temp.jpg"))
    ref_jpeg = extract_reference_image(jpeg)
    extracted_jpeg = extract_bits_from_difference(jpeg, nmi_interpolation(ref_jpeg))
    jpeg_text = bitstring_to_text(extracted_jpeg)
    correct_ratio_jpeg = sum(a == b for a, b in zip(jpeg_text, message)) / len(message) * 100
    #print(f"После JPEG извлечено: {jpeg_text}")
    print(f"Точность извлечения после сжатия: {correct_ratio_jpeg:.2f}%\n")

    noise = np.random.normal(0, 5, stego.shape).astype(np.int16)
    noisy = np.clip(stego.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    ref_noise = extract_reference_image(noisy)
    extracted_noise = extract_bits_from_difference(noisy, nmi_interpolation(ref_noise))
    extracted_text_noise = bitstring_to_text(extracted_noise)
    correct = sum(a == b for a, b in zip(extracted_text_noise, message))
    percent = correct / len(message) * 100
    #print(f"После добавления шума извлечено: {extracted_text_noise}")
    print(f"Точность извлечения после наложения шума: {percent:.2f}%")

    darker = np.clip(stego.astype(np.float32) * 0.9, 0, 255).astype(np.uint8)
    ref_dark = extract_reference_image(darker)
    extracted_dark = extract_bits_from_difference(darker, nmi_interpolation(ref_dark))
    extracted_text_dark = bitstring_to_text(extracted_dark)
    correct = sum(a == b for a, b in zip(extracted_text_dark, message))
    percent = correct / len(message) * 100
   # print(f"После уменьшения яркости извлечено: {extracted_text_dark}")
    print(f"Точность извлечения после уменьшения яркости: {percent:.2f}%")



def compare_histograms(img1, img2):
    print("Тест 4: Гистограммы")
    colors = ['r', 'g', 'b']
    fig, axs = plt.subplots(3, 2, figsize=(10, 6))
    for i in range(3):
        axs[i][0].hist(img1[..., i].ravel(), bins=256, color=colors[i])
        axs[i][0].set_title(f'Исходное - канал {colors[i]}')
        axs[i][1].hist(img2[..., i].ravel(), bins=256, color=colors[i])
        axs[i][1].set_title(f'После встраивания - канал {colors[i]}')
    plt.tight_layout()
    plt.show()


def run_all_tests(path):
    img = load_image(path)
    test_max_capacity(img)
    test_different_payloads(img)
    test_robustness(img)
    
    interpolated = nmi_interpolation(img)
    message = 1000 * "histogram_comparison_test"
    stego = embed(interpolated, text_to_bitstring(message))
    compare_histograms(interpolated, stego)

run_all_tests("C:/importantpapka/study/steganography/img/3.png")
