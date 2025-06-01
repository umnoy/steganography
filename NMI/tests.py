import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from embed import embed
from extraction import extract_bits_from_difference, extract_reference_image
from interpolation import nmi_interpolation
from extra import text_to_bitstring, bitstring_to_text, estimate_capacity, encode_length_and_data, decode_length_and_data
from skimage.metrics import peak_signal_noise_ratio as psnr
from metrics import ber

def load_image(path):
    return np.array(Image.open(path).convert("RGB"))

def save_image(array, path):
    Image.fromarray(array).save(path)

def test_max_capacity(img_arr):
    print("Тест 1: Максимальный объем встраивания")

    max_bits = estimate_capacity(img_arr)
    max_payload_bits = max_bits - 32
    max_chars = max_payload_bits // 8

    message = "f" * max_chars
    bitstring = text_to_bitstring(message)
    encoded = encode_length_and_data(bitstring)

    interpolated = nmi_interpolation(img_arr)
    stego = embed(interpolated, encoded)

    quality = psnr(interpolated, stego)

    raw_extracted = extract_bits_from_difference(stego, interpolated)
    clean_bits = decode_length_and_data(raw_extracted)
    extracted = bitstring_to_text(clean_bits)


    print(f"Макс. вместимость (байт): {max_bits // 8}")
    print(f"Встроено символов: {len(extracted)} из {len(message)}")
    print(f"PSNR: {quality:.2f} dB")
    print()

def test_different_payloads(img_arr):
    print("Тест 2: Разные объемы встраивания")
    for percent in [25, 50, 75, 100]:
        bits = estimate_capacity(img_arr) * percent // 100
        message = "f" * ((bits - 32) // 8)
        bitstring = text_to_bitstring(message)
        encoded = encode_length_and_data(bitstring)
        interpolated = nmi_interpolation(img_arr)
        stego = embed(interpolated, encoded)
        quality = psnr(interpolated, stego)
        raw_extracted = extract_bits_from_difference(stego, interpolated)
        ber_value = ber(encoded, raw_extracted)
        print(f"Загрузка: {percent}% | PSNR: {quality:.2f} dB | BER: {ber_value:.6f}") 
    print()

def test_robustness(img_arr):
    print("Тест 3: Робастность")
    message = 1000 * "robustness_test"
    bitstring = text_to_bitstring(message)
    interpolated = nmi_interpolation(img_arr)
    stego = embed(interpolated, bitstring)
    stego_img = Image.fromarray(stego)

    ref = extract_reference_image(stego)
    extracted = extract_bits_from_difference(stego, nmi_interpolation(ref))
    extracted_text = bitstring_to_text(extracted)
    ber_value = ber(bitstring, extracted)
    print(f"Было встроено 1000 раз сообщение: {message[:15]}")
    print(f"BER без сжатия: {ber_value:.6f}")

    blurred = np.array(stego_img.filter(ImageFilter.GaussianBlur(radius=1)))
    ref_blur = extract_reference_image(blurred)
    extracted_blur = extract_bits_from_difference(blurred, nmi_interpolation(ref_blur))
    ber_value_blur = ber(bitstring, extracted_blur)
    print(f"BER после размытия: {ber_value_blur:.6f}")

    rgb_stego_img = stego_img.convert("RGB")
    rgb_stego_img.save("temp.jpg", quality=85)
    jpeg = np.array(Image.open("temp.jpg"))
    ref_jpeg = extract_reference_image(jpeg)
    extracted_jpeg = extract_bits_from_difference(jpeg, nmi_interpolation(ref_jpeg))
    ber_value_jpeg = ber(bitstring, extracted_jpeg)
    print(f"BER после сжатия JPEG: {ber_value_jpeg:.6f}")

    noise = np.random.normal(0, 5, stego.shape).astype(np.int16)
    noisy = np.clip(stego.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    ref_noise = extract_reference_image(noisy)
    extracted_noise = extract_bits_from_difference(noisy, nmi_interpolation(ref_noise))
    ber_value_noise = ber(bitstring, extracted_noise)
    print(f"BER после наложения шума: {ber_value_noise:.6f}")

    darker = np.clip(stego.astype(np.float32) * 0.9, 0, 255).astype(np.uint8)
    ref_dark = extract_reference_image(darker)
    extracted_dark = extract_bits_from_difference(darker, nmi_interpolation(ref_dark))
    ber_value_dark = ber(bitstring, extracted_dark)
    print(f"BER после уменьшения яркости: {ber_value_dark:.6f}\n")

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
    message = 10000 * "histogram_comparison_test"
    stego = embed(interpolated, text_to_bitstring(message))
    compare_histograms(interpolated, stego)

run_all_tests("C:/importantpapka/study/steganography/NMI/img/1.png")