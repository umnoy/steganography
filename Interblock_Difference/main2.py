import cv2
import numpy as np
from scipy.fftpack import dctn, idctn
from numpy import isinf, isnan
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

BLOCK_SIZE = 8
WATERMARK_SIZE = 64
IMAGE_SIZE = 512
COEFF_COORDS = (1, 3)
T = 80
K = 12
Z = 2
ARNOLD_ITERATIONS = 5 

def apply_dct(block):
    return dctn(block, norm='ortho')

def apply_idct(block):
    return idctn(block, norm='ortho')

def arnold_transform(image, iterations):
    if image.shape[0] != image.shape[1]:
        raise ValueError("Входное изображение должно быть квадратным.")
    M = image.shape[0]
    arnold_image = np.copy(image)
    for _ in range(iterations):
        new_image = np.zeros_like(arnold_image)
        for i in range(M):
            for j in range(M):
                j_prime = (i + 2 * j) % M
                i_prime = (i + j) % M
                new_image[i_prime, j_prime] = arnold_image[i, j]
        arnold_image = new_image
    return arnold_image

def inverse_arnold_transform(image, iterations):
    if image.shape[0] != image.shape[1]:
        raise ValueError("Входное изображение должно быть квадратным.")
    M = image.shape[0]
    inv_arnold_image = np.copy(image)
    for _ in range(iterations):
        new_image = np.zeros_like(inv_arnold_image)
        for i_prime in range(M):
            for j_prime in range(M):
                i = (2 * i_prime - j_prime) % M
                j = (-i_prime + j_prime) % M
                new_image[i, j] = inv_arnold_image[i_prime, j_prime]
        inv_arnold_image = new_image
    return inv_arnold_image

def calculate_median_ac(dct_block):
    zigzag_coords = [(0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1), (3, 0)]
    ac_coeffs = [dct_block[y, x] for y, x in zigzag_coords]
    if not ac_coeffs:
        return 0.0
    return np.median(ac_coeffs)

def calculate_M(dct_block, Z):
    dc_coeff = dct_block[0, 0]
    median_ac = calculate_median_ac(dct_block)
    if isinf(dc_coeff) or isnan(dc_coeff) or abs(dc_coeff) < 1e-9:
        return 1e-9
    if abs(dc_coeff) > 1000 or abs(dc_coeff) < 1:
        if isinf(median_ac) or isnan(median_ac):
            return 1e-9
        return abs(Z * median_ac)
    else:
        diff_dc_median = dc_coeff - median_ac
        if isinf(diff_dc_median) or isnan(diff_dc_median):
            return 1e-9
        return abs(Z * diff_dc_median / dc_coeff)

def embed_bit_modified_logic(block1_dct, block2_dct, watermark_bit, coeff_coords, M, T, K):
    C1 = block1_dct[coeff_coords]
    C2 = block2_dct[coeff_coords]
    delta = C2 - C1
    modified_block2_dct = block2_dct.copy()
    C2_modified = modified_block2_dct[coeff_coords]
    if isinf(M) or isnan(M) or M < 1e-9:
        return block2_dct.copy()
    if watermark_bit == 1:
        if delta > T - K:
            while delta > T - K:
                C2_modified -= M
                delta = C2_modified - C1
        elif K > delta > -T/2:
            while delta < K:
                C2_modified += M
                delta = C2_modified - C1
        elif delta < -T/2:
            while delta > -T - K:
                C2_modified -= M
                delta = C2_modified - C1
    elif watermark_bit == 0:
        if delta > T/2:
            while delta <= T + K:
                C2_modified += M
                delta = C2_modified - C1
        elif -K < delta < T/2:
            while delta >= -K:
                C2_modified -= M
                delta = C2_modified - C1
        elif delta < K - T:
            while delta <= K - T:
                C2_modified += M
                delta = C2_modified - C1
    modified_block2_dct[coeff_coords] = C2_modified
    return modified_block2_dct

def postprocess_dct_block(modified_dct_block):
    pixel_block_float = idctn(modified_dct_block, norm='ortho')
    pixel_block_float[isinf(pixel_block_float) | isnan(pixel_block_float)] = 0
    processed_block_pixels = np.round(pixel_block_float + 128).clip(0, 255).astype(np.uint8)
    return processed_block_pixels

def extract_single_bit_from_pair_robust(block1_dct, block2_dct, coeff_coords, T, K):
    C1 = block1_dct[coeff_coords]
    C2 = block2_dct[coeff_coords]
    delta = C2 - C1
    if isinf(delta) or isnan(delta):
        return None
    epsilon = 1e-9
    if abs(delta - T) < epsilon: return 1
    elif abs(delta + T) < epsilon: return 0
    elif abs(delta) < epsilon: return 0
    elif abs(delta - K) < epsilon and K > 0: return 0
    elif abs(delta + K) < epsilon and K > 0: return 1
    if (delta > T) or (-T < delta < 0):
        return 1
    elif (delta < -T) or (0 < delta < T):
        return 0
    else:
        return None

def embed_watermark_blue_channel_lr(image_path, watermark_path, T, K, Z, coeff_coords, block_size):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Ошибка: Изображение не найдено по пути: {image_path}")
        return None
    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    if watermark is None:
        print(f"Ошибка: Водяной знак не найден по пути: {watermark_path}")
        return None
    
    # Применяем преобразование Арнольда к водяному знаку
    watermark_transformed = arnold_transform(watermark, ARNOLD_ITERATIONS)
    
    num_blocks_x = IMAGE_SIZE // block_size
    num_blocks_y = IMAGE_SIZE // block_size
    expected_wm_len = num_blocks_y * (num_blocks_x - 1)
    watermark_flat = (watermark_transformed > 128).astype(np.uint8).flatten()
    if len(watermark_flat) < expected_wm_len:
        watermark_to_embed = watermark_flat
    elif len(watermark_flat) > expected_wm_len:
        watermark_to_embed = watermark_flat[:expected_wm_len]
    else:
        watermark_to_embed = watermark_flat
    if img.shape[0] != IMAGE_SIZE or img.shape[1] != IMAGE_SIZE:
        print(f"Ошибка: Размер изображения должен быть {IMAGE_SIZE}x{IMAGE_SIZE}")
        return None
    b_channel, g_channel, r_channel = cv2.split(img)
    b_channel_float = b_channel.astype(np.float32) - 128
    dct_blocks = []
    for i in range(num_blocks_y):
        row_of_blocks = []
        for j in range(num_blocks_x):
            block = b_channel_float[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            dct_block = apply_dct(block)
            row_of_blocks.append(dct_block)
        dct_blocks.append(row_of_blocks)
    watermark_index = 0
    for i in range(num_blocks_y):
        for j in range(num_blocks_x - 1):
            if watermark_index < len(watermark_to_embed):
                block1_dct = dct_blocks[i][j]
                block2_dct = dct_blocks[i][j+1]
                M = calculate_M(block1_dct, Z)
                bit_to_embed = watermark_to_embed[watermark_index]
                modified_block2_dct = embed_bit_modified_logic(block1_dct, block2_dct, bit_to_embed, coeff_coords, M, T, K)
                dct_blocks[i][j+1] = modified_block2_dct
                watermark_index += 1
            else:
                break
        if watermark_index == len(watermark_to_embed):
            break
    b_channel_watermarked = np.zeros_like(b_channel, dtype=np.uint8)
    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            processed_block_pixels = postprocess_dct_block(dct_blocks[i][j])
            b_channel_watermarked[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = processed_block_pixels
    watermarked_img = cv2.merge((b_channel_watermarked, g_channel, r_channel))
    
    psnr_value = psnr(img, watermarked_img, data_range=255)
    ssim_value, _ = ssim(img, watermarked_img, data_range=255, multichannel=True, channel_axis=2, full=True)
    print(f"Показатели качества встраивания:")
    print(f"PSNR: {psnr_value:.2f} дБ")
    print(f"SSIM: {ssim_value:.4f}")
    
    return watermarked_img

def extract_watermark_blue_channel_lr_64x64_dominant_pad_fixed(watermarked_image_path, T, K, coeff_coords, block_size):
    img = cv2.imread(watermarked_image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Ошибка: Изображение не найдено по пути: {watermarked_image_path}")
        return None
    if img.shape[0] != IMAGE_SIZE or img.shape[1] != IMAGE_SIZE:
        print(f"Ошибка: Размер изображения должен быть {IMAGE_SIZE}x{IMAGE_SIZE}")
        return None
    b_channel, _, _ = cv2.split(img)
    b_channel_float = b_channel.astype(np.float32) - 128
    dct_blocks = []
    num_blocks_x = IMAGE_SIZE // block_size
    num_blocks_y = IMAGE_SIZE // block_size
    for i in range(num_blocks_y):
        row_of_blocks = []
        for j in range(num_blocks_x):
            block = b_channel_float[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            dct_block = apply_dct(block)
            row_of_blocks.append(dct_block)
        dct_blocks.append(row_of_blocks)
    extracted_bits_reliable = []
    extracted_bits_all = []
    for i in range(num_blocks_y):
        for j in range(num_blocks_x - 1):
            block1_dct = dct_blocks[i][j]
            block2_dct = dct_blocks[i][j+1]
            extracted_bit = extract_single_bit_from_pair_robust(block1_dct, block2_dct, coeff_coords, T, K)
            if extracted_bit is not None:
                extracted_bits_reliable.append(extracted_bit)
                extracted_bits_all.append(extracted_bit)
            else:
                extracted_bits_all.append(0)
    actual_extracted_shape = (num_blocks_y - 1, num_blocks_x)
    if len(extracted_bits_all) != actual_extracted_shape[0] * actual_extracted_shape[1]:
        print(f"Ошибка: Количество извлеченных битов ({len(extracted_bits_all)}) не соответствует ожидаемому размеру ({actual_extracted_shape[0] * actual_extracted_shape[1]}) для формы {actual_extracted_shape}.")
        return np.array(extracted_bits_all)
    extracted_watermark_actual = np.array(extracted_bits_all).reshape(63, 64)
    if len(extracted_bits_reliable) > 0:
        count_ones = np.sum(extracted_bits_reliable)
        total_reliable_bits = len(extracted_bits_reliable)
        count_zeros = total_reliable_bits - count_ones
        if count_ones > count_zeros:
            dominant_color = 1
        elif count_zeros > count_ones:
            dominant_color = 0
        else:
            dominant_color = 0
    else:
        dominant_color = 0
    #print(f"Доминирующий бит в надежно извлеченных данных: {dominant_color}")
    target_shape = (WATERMARK_SIZE, WATERMARK_SIZE)
    if extracted_watermark_actual.shape != target_shape:
        padding_needed = ((0, target_shape[0] - extracted_watermark_actual.shape[0]), (0, 0))
        extracted_watermark_padded = np.pad(extracted_watermark_actual, padding_needed, mode='constant', constant_values=dominant_color)
    else:
        extracted_watermark_padded = extracted_watermark_actual
    
    extracted_watermark_restored = inverse_arnold_transform(extracted_watermark_padded, ARNOLD_ITERATIONS)
    inverted_extracted_wm = 1 - extracted_watermark_restored
    return inverted_extracted_wm

def main():
    print("Выберите режим:")
    print("1 - Встраивание водяного знака")
    print("2 - Извлечение водяного знака")
    while True:
        try:
            choice = int(input("Введите ваш выбор (1 или 2): "))
            if choice in [1, 2]:
                break
            else:
                print("Неверный выбор. Пожалуйста, введите 1 или 2.")
        except ValueError:
            print("Неверный ввод. Пожалуйста, введите число (1 или 2).")
    
    if choice == 1:
        image_path = input("Введите путь к изображению-контейнеру: ")
        watermark_path = input("Введите путь к изображению водяного знака: ")
        watermarked_image = embed_watermark_blue_channel_lr(
            image_path, watermark_path, T, K, Z, COEFF_COORDS, BLOCK_SIZE
        )
        if watermarked_image is not None:
            cv2.imwrite('watermarked_image.png', watermarked_image)
            print("Изображение с водяным знаком сохранено как 'watermarked_image.png'")
    else:
        watermarked_image_path = input("Введите путь к изображению с водяным знаком: ")
        extracted_wm = extract_watermark_blue_channel_lr_64x64_dominant_pad_fixed(
            watermarked_image_path, T, K, COEFF_COORDS, BLOCK_SIZE
        )
        inverted_extracted_wm =  extracted_wm
        if extracted_wm is not None:
            print("\nИзвлеченный водяной знак:")
            print(inverted_extracted_wm)
            if extracted_wm.ndim == 2:
                if extracted_wm.shape[0] > 0 and extracted_wm.shape[1] > 0:
                    cv2.imwrite('extracted_watermark.png', inverted_extracted_wm.astype(np.uint8) * 255)
                    print(f"Извлеченный водяной знак сохранен как 'extracted_watermark.png' с формой {extracted_wm.shape}")
                else:
                    print("Извлеченный водяной знак пустой или имеет нулевой размер, не может быть сохранен как изображение.")
            else:
                print("Извлеченный водяной знак не является 2D массивом и не может быть сохранен как изображение.")

if __name__ == "__main__":
    main()