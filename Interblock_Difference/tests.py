import cv2
import numpy as np
import os
import sys
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from main3 import embed_watermark_blue_channel_lr, extract_watermark_blue_channel_lr_64x64_dominant_pad_fixed
import itertools
import matplotlib.pyplot as plt

IMAGE_SIZE = 512
BLOCK_SIZE = 8
WATERMARK_SIZE = 64
TESTS_DIR = "tests"

# Класс для перенаправления вывода print в файл и консоль
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # Обеспечивает немедленную запись

    def flush(self):
        for f in self.files:
            f.flush()

def create_tests_directory(dir_path=TESTS_DIR):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

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

def generate_chart(attack_name, labels, ssim_values, ber_values):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = np.arange(len(labels))
    width = 0.35

    # График SSIM
    ax1.bar(x - width/2, ssim_values, width, label='SSIM', color='skyblue')
    ax1.set_ylabel('SSIM', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(-1, 1)  # Диапазон для SSIM [-1, 1]

    # График BER
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, ber_values, width, label='BER', color='salmon')
    ax2.set_ylabel('BER', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 0.5)  # Ограничиваем ось BER до 0.5

    # Настройки осей
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_xlabel('Parameter Combinations')
    ax1.set_title(f'SSIM and BER vs. Parameter Combinations ({attack_name})')

    # Легенда
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    # Сохранение графика
    plt.tight_layout()
    chart_path = os.path.join(TESTS_DIR, f"metrics_chart_{attack_name}.png")
    plt.savefig(chart_path)
    plt.close()
    return chart_path

def run_tests(image_path, watermark_path):
    create_tests_directory()
    original_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    original_wm = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    if original_img is None or original_wm is None:
        print("Ошибка: Не удалось загрузить изображение или водяной знак.")
        return
    original_wm = (original_wm > 128).astype(np.uint8)

    # Define parameter sets
    COEFF_COORDS_LIST = [(1, 3), (0, 1)]
    T_LIST = [80, 150]
    K_LIST = [12, 20]
    Z_LIST = [2, 7]

    # Generate all combinations
    param_combinations = list(itertools.product(COEFF_COORDS_LIST, T_LIST, K_LIST, Z_LIST))

    # Store metrics for all attacks
    attack_metrics = {}
    for idx, (coeff_coords, T, K, Z) in enumerate(param_combinations):
        param_label = f"CC{coeff_coords[0]}{coeff_coords[1]}_T{T}_K{K}_Z{Z}"
        print(f"\nТестирование комбинации {idx + 1}/{len(param_combinations)}: "
              f"COEFF_COORDS={coeff_coords}, T={T}, K={K}, Z={Z}")

        # Create a subdirectory for this parameter combination
        param_dir = os.path.join(TESTS_DIR, f"params_{idx}_{param_label}")
        create_tests_directory(param_dir)

        # Встраивание
        watermarked_img = embed_watermark_blue_channel_lr(
            image_path, watermark_path, T, K, Z, coeff_coords, BLOCK_SIZE
        )
        if watermarked_img is None:
            print("Ошибка: Не удалось встроить водяной знак.")
            continue
        watermarked_path = os.path.join(param_dir, "watermarked.png")
        cv2.imwrite(watermarked_path, watermarked_img)

        # Оценка незаметности
        psnr_value = psnr(original_img, watermarked_img, data_range=255)
        ssim_value, _ = ssim(original_img, watermarked_img, data_range=255, multichannel=True, channel_axis=2, full=True)
        print(f"Незаметность встраивания:")
        print(f"PSNR: {psnr_value:.2f} дБ")
        print(f"SSIM: {ssim_value:.4f}")

        # Тестирование робастности
        test_cases = [
            ("no_attack", lambda x, p: (cv2.imwrite(p, x), x)[1], watermarked_img, ".png"),
            ("jpeg_q90", lambda x, p: apply_jpeg_compression(x, 90, p), watermarked_img, ".jpg"),
            ("jpeg_q50", lambda x, p: apply_jpeg_compression(x, 50, p), watermarked_img, ".jpg"),
            ("jpeg_q10", lambda x, p: apply_jpeg_compression(x, 10, p), watermarked_img, ".jpg"),
            ("jpeg_q5", lambda x, p: apply_jpeg_compression(x, 5, p), watermarked_img, ".jpg"),
            ("center_crop_128", lambda x, p: apply_center_crop(x, 128, p), watermarked_img, ".png"),
            ("rotation_5deg", lambda x, p: apply_rotation(x, 5, p), watermarked_img, ".png"),
            ("brightness_plus50", lambda x, p: apply_brightness_adjustment(x, 50, p), watermarked_img, ".png"),
            ("crop_50px", lambda x, p: apply_cropping(x, 50, p), watermarked_img, ".png"),
            ("gaussian_blur", lambda x, p: apply_gaussian_blur(x, 5, 1.0, p), watermarked_img, ".png"),
            ("gaussian_noise", lambda x, p: apply_gaussian_noise(x, 0, 25, p), watermarked_img, ".png"),
            ("heavy_noise", lambda x, p: apply_gaussian_noise(x, 0, 100, p), watermarked_img, ".png")
        ]

        for test_name, attack_func, input_img, extension in test_cases:
            attacked_path = os.path.join(param_dir, f"attacked_{test_name}{extension}")
            attacked_img = attack_func(input_img, attacked_path)
            if attacked_img is None:
                print(f"Ошибка: Не удалось применить атаку '{test_name}' (невалидное изображение).")
                continue
            attacked_img = cv2.imread(attacked_path, cv2.IMREAD_COLOR) if test_name != "no_attack" else attacked_img
            if attacked_img is None or attacked_img.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
                print(f"Ошибка: Не удалось загрузить атакованное изображение по пути: {attacked_path} или неверный размер.")
                continue
            ssim_attack, _ = ssim(original_img, attacked_img, data_range=255, multichannel=True, channel_axis=2, full=True)
            print(f"\nАтака: {test_name}")
            print("Характеристики извлечения:")
            print(f"-SSIM атакованного изображения: {ssim_attack:.4f};")
            
            extracted_wm = extract_watermark_blue_channel_lr_64x64_dominant_pad_fixed(
                attacked_path, T, K, coeff_coords, BLOCK_SIZE
            )
            if extracted_wm is None:
                print(f"Ошибка: Не удалось извлечь водяной знак для атаки '{test_name}'.")
                continue
            extracted_wm_path = os.path.join(param_dir, f"extracted_wm_{test_name}.png")
            cv2.imwrite(extracted_wm_path, (extracted_wm).astype(np.uint8) * 255)
            
            ber = calculate_bit_error_rate(original_wm, 1 - extracted_wm)
            # Вычисляем SSIM между исходным и извлечённым водяным знаком
            ssim_wm, _ = ssim(original_wm, extracted_wm, data_range=1, full=True)
            print(f"-BER (доля ошибок): {ber:.4f};")
            print(f"-SSIM между исходным и извлечённым водяным знаком: {ssim_wm:.4f}.")
            print(f"Извлеченный водяной знак сохранен как '{extracted_wm_path}'")

            # Store metrics for each attack
            if test_name not in attack_metrics:
                attack_metrics[test_name] = {"labels": [], "ssim_values": [], "ber_values": []}
            attack_metrics[test_name]["labels"].append(param_label)
            attack_metrics[test_name]["ssim_values"].append(float(ssim_attack))
            attack_metrics[test_name]["ber_values"].append(float(ber))

    # Generate charts for all attacks
    for attack_name, metrics in attack_metrics.items():
        chart_path = generate_chart(attack_name, metrics["labels"], metrics["ssim_values"], metrics["ber_values"])
        print(f"\nГрафик для атаки '{attack_name}' сохранен как '{chart_path}'")

def main():
    create_tests_directory()
    
    log_file_path = os.path.join(TESTS_DIR, "log.txt")
    log_file = open(log_file_path, "w", encoding="utf-8")
    
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    print("Тестирование алгоритма встраивания водяного знака")
    image_path = input("Введите путь к изображению-контейнеру: ")
    watermark_path = input("Введите путь к изображению водяного знака: ")
    run_tests(image_path, watermark_path)

    sys.stdout = original_stdout
    log_file.close()

if __name__ == "__main__":
    main()