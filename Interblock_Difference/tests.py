import cv2
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from main2 import embed_watermark_blue_channel_lr, extract_watermark_blue_channel_lr_64x64_dominant_pad_fixed

IMAGE_SIZE = 512
BLOCK_SIZE = 8
WATERMARK_SIZE = 64
COEFF_COORDS = (0, 2)
T = 80
K = 12
Z = 2
TESTS_DIR = "tests"

def create_tests_directory():
    if not os.path.exists(TESTS_DIR):
        os.makedirs(TESTS_DIR)

def apply_jpeg_compression(image, quality, output_path):
    success = cv2.imwrite(output_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return None
    return cv2.imread(output_path, cv2.IMREAD_COLOR)

def apply_center_crop(image, crop_size, output_path):
    h, w = image.shape[:2]
    start_x = (w - crop_size) // 2
    start_y = (h - crop_size) // 2
    if start_x < 0 or start_y < 0 or start_x + crop_size > w or start_y + crop_size > h:
        return None
    cropped = image.copy()
    cropped[start_y:start_y + crop_size, start_x:start_x + crop_size] = 0
    success = cv2.imwrite(output_path, cropped)
    if not success:
        return None
    return cropped

def apply_rotation(image, angle, output_path):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    success = cv2.imwrite(output_path, rotated)
    if not success:
        return None
    return rotated

def apply_brightness_adjustment(image, factor, output_path):
    adjusted = cv2.convertScaleAbs(image, beta=factor)
    success = cv2.imwrite(output_path, adjusted)
    if not success:
        return None
    return adjusted

def apply_cropping(image, crop_pixels, output_path):
    h, w = image.shape[:2]
    if crop_pixels * 2 >= h or crop_pixels * 2 >= w:
        return None
    cropped = image[crop_pixels:-crop_pixels, crop_pixels:-crop_pixels]
    resized = cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    success = cv2.imwrite(output_path, resized)
    if not success:
        return None
    return resized

def apply_gaussian_blur(image, kernel_size, sigma, output_path):
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    success = cv2.imwrite(output_path, blurred)
    if not success:
        return None
    return blurred

def apply_gaussian_noise(image, mean, sigma, output_path):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    success = cv2.imwrite(output_path, noisy)
    if not success:
        return None
    return noisy

def calculate_bit_error_rate(original_wm, extracted_wm):
    inverted_original_wm = 1 - original_wm
    if inverted_original_wm.shape != extracted_wm.shape:
        inverted_original_wm = inverted_original_wm[:extracted_wm.shape[0], :extracted_wm.shape[1]]
    total_bits = inverted_original_wm.size
    errors = np.sum(inverted_original_wm != extracted_wm)
    return errors / total_bits if total_bits > 0 else 1.0

def run_tests(image_path, watermark_path):
    create_tests_directory()
    original_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    original_wm = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    if original_img is None or original_wm is None:
        print("Ошибка: Не удалось загрузить изображение или водяной знак.")
        return
    original_wm = (original_wm > 128).astype(np.uint8)

    print(f"\nТестирование с параметрами T={T}, K={K}")

    # Встраивание
    watermarked_img = embed_watermark_blue_channel_lr(
        image_path, watermark_path, T, K, Z, COEFF_COORDS, BLOCK_SIZE
    )
    if watermarked_img is None:
        print("Ошибка: Не удалось встроить водяной знак.")
        return
    watermarked_path = os.path.join(TESTS_DIR, "watermarked.png")
    cv2.imwrite(watermarked_path, watermarked_img)

    # Оценка незаметности
    psnr_value = psnr(original_img, watermarked_img, data_range=255)
    ssim_value, _ = ssim(original_img, watermarked_img, data_range=255, multichannel=True, channel_axis=2, full=True)
    print(f"Незаметность встраивания:")
    print(f"PSNR: {psnr_value:.2f} дБ")
    print(f"SSIM: {ssim_value:.4f}")

    # Тестирование робастности
    test_cases = [
        ("no_attack", lambda x, p: (cv2.imwrite(p, x), x)[1], watermarked_img),
        ("jpeg_q90", lambda x, p: apply_jpeg_compression(x, 90, p), watermarked_img),
        ("jpeg_q50", lambda x, p: apply_jpeg_compression(x, 50, p), watermarked_img),
        ("jpeg_q10", lambda x, p: apply_jpeg_compression(x, 10, p), watermarked_img),
        ("center_crop_128", lambda x, p: apply_center_crop(x, 128, p), watermarked_img),
        ("rotation_5deg", lambda x, p: apply_rotation(x, 5, p), watermarked_img),
        ("brightness_plus50", lambda x, p: apply_brightness_adjustment(x, 50, p), watermarked_img),
        ("crop_50px", lambda x, p: apply_cropping(x, 50, p), watermarked_img),
        ("gaussian_blur", lambda x, p: apply_gaussian_blur(x, 5, 1.0, p), watermarked_img),
        ("gaussian_noise", lambda x, p: apply_gaussian_noise(x, 0, 25, p), watermarked_img)
    ]

    for test_name, attack_func, input_img in test_cases:
        attacked_path = os.path.join(TESTS_DIR, f"attacked_{test_name}.png")
        attacked_img = attack_func(input_img, attacked_path)
        if attacked_img is None:
            print(f"Ошибка: Не удалось применить атаку '{test_name}' (невалидное изображение).")
            continue
        attacked_img = cv2.imread(attacked_path, cv2.IMREAD_COLOR) if test_name != "no_attack" else attacked_img
        if attacked_img is None or attacked_img.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
            print(f"Ошибка: Не удалось загрузить атакованное изображение по пути: {attacked_path} или неверный размер.")
            continue
        psnr_attack = psnr(original_img, attacked_img, data_range=255)
        print(f"\nАтака: {test_name}")
        print(f"PSNR атакованного изображения: {psnr_attack:.2f} дБ")
        
        extracted_wm = extract_watermark_blue_channel_lr_64x64_dominant_pad_fixed(
            attacked_path, T, K, COEFF_COORDS, BLOCK_SIZE
        )
        if extracted_wm is None:
            print(f"Ошибка: Не удалось извлечь водяной знак для атаки '{test_name}'.")
            continue
        extracted_wm_path = os.path.join(TESTS_DIR, f"extracted_wm_{test_name}.png")
        cv2.imwrite(extracted_wm_path, (1 - extracted_wm).astype(np.uint8) * 255)
        
        ber = calculate_bit_error_rate(original_wm, extracted_wm)
        print(f"BER (доля ошибок): {ber:.4f}")
        print(f"Извлеченный водяной знак сохранен как '{extracted_wm_path}'")

def main():
    print("Тестирование алгоритма встраивания водяного знака")
    image_path = input("Введите путь к изображению-контейнеру: ")
    watermark_path = input("Введите путь к изображению водяного знака: ")
    run_tests(image_path, watermark_path)

if __name__ == "__main__":
    main()