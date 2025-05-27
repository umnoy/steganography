import cv2
import numpy as np
from scipy.fftpack import dctn, idctn
from numpy import isinf, isnan

# --- Параметры алгоритма (должны быть одинаковыми для встраивания и извлечения) ---
BLOCK_SIZE = 8
WATERMARK_SIZE = 64 # Ожидаемый размер водяного знака (одна сторона)
IMAGE_SIZE = 512
COEFF_COORDS = (0, 2) # Координаты (1,3) из статьи в 0-based indexing
T = 80
K = 12
Z = 2 # Параметр для расчета M
# Направление обхода фиксировано: LR
# --- Конец параметров ---


# --- Вспомогательные функции ---
# (Остаются без изменений, так как работают с отдельными парами блоков)

def apply_dct(block):
    """Применяет 2D ДКП к блоку."""
    return dctn(block, norm='ortho')

def apply_idct(block):
    """Применяет 2D обратное ДКП к блоку."""
    return idctn(block, norm='ortho')

def arnold_transform(image, iterations):
    """Применяет преобразование Арнольда к квадратной матрице."""
    if image.shape[0] != image.shape[1]:
        raise ValueError("Input image must be square.")
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
    return new_image

def inverse_arnold_transform(image, iterations):
    """Применяет обратное преобразование Арнольда к квадратной матрице."""
    if image.shape[0] != image.shape[1]:
        raise ValueError("Input image must be square.")
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
    """Вычисляет медиану первых 9 АС-коэффициентов по зигзагу."""
    zigzag_coords = [(0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1), (3, 0)]
    ac_coeffs = [dct_block[y, x] for y, x in zigzag_coords]
    if not ac_coeffs:
        return 0.0
    return np.median(ac_coeffs)

def calculate_M(dct_block, Z):
    """Вычисляет силу модификации M для блока Block1."""
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
    """
    Встраивает один бит ЦВЗ в пару блоков путем модификации коэффициента
    ВТОРОГО блока на основе разности C2-C1.
    Возвращает модифицированный block2_dct.
    """
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
    """
    Применяет обратное ДКП (ОДКП), добавляет 128, округляет и клиппирует
    значения блока ДКП, чтобы получить пиксельные значения в диапазоне [0, 255].
    """
    pixel_block_float = idctn(modified_dct_block, norm='ortho')
    pixel_block_float[isinf(pixel_block_float) | isnan(pixel_block_float)] = 0

    processed_block_pixels = np.round(pixel_block_float + 128).clip(0, 255).astype(np.uint8)

    return processed_block_pixels

def extract_single_bit_from_pair_robust(block1_dct, block2_dct, coeff_coords, T, K):
    """
    Извлекает один бит ЦВЗ из пары ДКП блоков на основе разности коэффициентов.
    Учитывает модифицированную логику встраивания (разность C2-C1, модификация C2)
    и обрабатывает граничные случаи и случаи с inf/NaN.
    """
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


# --- Функции встраивания и извлечения ЦВЗ в синий канал (только LR) ---

def embed_watermark_blue_channel_lr(image_path, watermark_path, T, K, Z, coeff_coords, block_size):
    """
    Встраивает ЦВЗ в синий канал изображения-контейнера по первому методу
    с модифицированной логикой встраивания, используя только обход LR.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Ошибка: Изображение не найдено по пути: {image_path}")
        return None

    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    #watermark = arnold_transform(watermark, 5)
    if watermark is None:
         print(f"Ошибка: ЦВЗ не найден по пути: {watermark_path}")
         return None

    num_blocks_x = IMAGE_SIZE // block_size
    num_blocks_y = IMAGE_SIZE // block_size
    # Для LR направления емкость фиксирована
    expected_wm_len = num_blocks_y * (num_blocks_x - 1) # 64 * 63 = 4032 бита

    watermark_flat = (watermark > 128).astype(np.uint8).flatten()

    if len(watermark_flat) < expected_wm_len:
         print(f"Предупреждение: Размер ЦВЗ ({len(watermark_flat)} битов) меньше емкости ({expected_wm_len} битов) для LR. Будут встроены только первые {len(watermark_flat)} битов.")
         watermark_to_embed = watermark_flat
    elif len(watermark_flat) > expected_wm_len:
        print(f"Предупреждение: Размер ЦВЗ ({len(watermark_flat)} битов) больше емкости ({expected_wm_len} битов) для LR. Будут встроены только первые {expected_wm_len} битов.")
        watermark_to_embed = watermark_flat[:expected_wm_len]
    else:
        watermark_to_embed = watermark_flat # Размер совпадает с емкостью 4032

    if img.shape[0] != IMAGE_SIZE or img.shape[1] != IMAGE_SIZE:
         print(f"Ошибка: Размер изображения должен быть {IMAGE_SIZE}x{IMAGE_SIZE}")
         return None

    b_channel, g_channel, r_channel = cv2.split(img)
    b_channel_float = b_channel.astype(np.float32) - 128

    # 1. Применяем ДКП ко ВСЕМ блокам синего канала
    dct_blocks = []
    for i in range(num_blocks_y):
        row_of_blocks = []
        for j in range(num_blocks_x):
            block = b_channel_float[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            dct_block = apply_dct(block)
            row_of_blocks.append(dct_block)
        dct_blocks.append(row_of_blocks)

    # 2. Проходим по парам блоков согласно направлению LR и встраиваем биты
    watermark_index = 0

    # Только LR обход
    for i in range(num_blocks_y):
        for j in range(num_blocks_x - 1):
            if watermark_index < len(watermark_to_embed):
                block1_dct = dct_blocks[i][j]
                block2_dct = dct_blocks[i][j+1]

                # Вычисляем M для Block1
                M = calculate_M(block1_dct, Z)

                # Получаем бит ЦВЗ
                bit_to_embed = watermark_to_embed[watermark_index]

                # Встраиваем бит - модифицируется block2_dct
                modified_block2_dct = embed_bit_modified_logic(block1_dct, block2_dct, bit_to_embed, coeff_coords, M, T, K)

                # Обновляем block2_dct в списке блоков
                dct_blocks[i][j+1] = modified_block2_dct

                watermark_index += 1
            else:
                break
        if watermark_index == len(watermark_to_embed):
            break

    print(f"Всего встроено {watermark_index} битов.")

    # 3. Применяем ОДКП и постобработку ко ВСЕМ блокам ДКП
    b_channel_watermarked = np.zeros_like(b_channel, dtype=np.uint8)

    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            processed_block_pixels = postprocess_dct_block(dct_blocks[i][j])
            b_channel_watermarked[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = processed_block_pixels

    watermarked_img = cv2.merge((b_channel_watermarked, g_channel, r_channel))

    return watermarked_img


def extract_watermark_blue_channel_lr_64x64_dominant_pad_fixed(watermarked_image_path, T, K, coeff_coords, block_size):
    """
    Извлекает ЦВЗ из синего канала изображения с водяным знаком по первому методу
    с модифицированной логикой встраивания и обработкой границ при извлечении,
    используя только обход LR.
    Собирает извлеченные биты в матрицу 63x64 (как показал опыт)
    и дополняет ее до размера 64x64, используя доминирующий бит.
    """
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
    num_blocks_x = IMAGE_SIZE // block_size # 64
    num_blocks_y = IMAGE_SIZE // block_size # 64

    for i in range(num_blocks_y):
        row_of_blocks = []
        for j in range(num_blocks_x):
            block = b_channel_float[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            dct_block = apply_dct(block)
            row_of_blocks.append(dct_block)
        dct_blocks.append(row_of_blocks)

    extracted_bits_reliable = [] # Список только надежно извлеченных битов
    extracted_bits_all = [] # Список всех битов, включая 0 для неопределенных

    # Только LR обход
    for i in range(num_blocks_y): # 64 строки
        for j in range(num_blocks_x - 1): # 63 колонки пар (0-62)
            block1_dct = dct_blocks[i][j]
            block2_dct = dct_blocks[i][j+1] # Блок справа
            # Извлекаем бит из пары
            extracted_bit = extract_single_bit_from_pair_robust(block1_dct, block2_dct, coeff_coords, T, K)

            if extracted_bit is not None:
                 extracted_bits_reliable.append(extracted_bit)
                 extracted_bits_all.append(extracted_bit)
            else:
                 extracted_bits_all.append(0) # Добавляем 0 для неопределенных битов

    print(f"Всего извлечено (включая неопределенные): {len(extracted_bits_all)} битов.")
    print(f"Надежно извлечено: {len(extracted_bits_reliable)} битов.")

    # --- Формируем матрицу извлеченного ЦВЗ фактически извлеченного размера (63x64) ---
    # Используем форму (63, 64), так как опытным путем установлено, что это предотвращает скос.
    actual_extracted_shape = (num_blocks_y - 1, num_blocks_x) # 63x64 - Это форма, которая визуально корректна
    # Проверяем, что количество битов соответствует этой форме
    if len(extracted_bits_all) != actual_extracted_shape[0] * actual_extracted_shape[1]:
         print(f"Ошибка: Количество извлеченных битов ({len(extracted_bits_all)}) не соответствует ожидаемому размеру ({actual_extracted_shape[0] * actual_extracted_shape[1]}) для формы {actual_extracted_shape}.")
         return np.array(extracted_bits_all) # Возвращаем 1D массив в случае ошибки

    extracted_watermark_actual = np.array(extracted_bits_all).reshape(63,64) # Reshape в 63x64

    print(f"Фактическая форма извлеченного ЦВЗ (для визуального восстановления): {extracted_watermark_actual.shape}")

    # --- Вычисляем доминирующий бит в фактически извлеченных данных (63x64) ---
    if len(extracted_bits_reliable) > 0:
        count_ones = np.sum(extracted_bits_reliable)
        total_reliable_bits = len(extracted_bits_reliable)
        count_zeros = total_reliable_bits - count_ones

        if count_ones > count_zeros:
            dominant_color = 1
        elif count_zeros > count_ones:
            dominant_color = 0
        else:
            dominant_color = 0 # В случае равенства
    else:
         dominant_color = 0 # Если надежных битов нет

    print(f"Доминирующий бит в надежно извлеченных данных: {dominant_color}")

    # --- Дополняем матрицу 63x64 до 64x64 с использованием доминирующего цвета ---
    target_shape = (WATERMARK_SIZE, WATERMARK_SIZE) # 64x64

    if extracted_watermark_actual.shape != target_shape:
        # Текущая форма (63, 64), целевая (64, 64). Нужно добавить 1 строку снизу.
        # padding = ((top_rows, bottom_rows), (left_cols, right_cols))
        padding_needed = ((0, target_shape[0] - extracted_watermark_actual.shape[0]), (0, 0)) # Добавляем 1 строку снизу
        extracted_watermark_padded = np.pad(extracted_watermark_actual, padding_needed, mode='constant', constant_values=dominant_color)
        print(f"Извлеченный ЦВЗ дополнен до формы {extracted_watermark_padded.shape} цветом {dominant_color}")
    else:
        extracted_watermark_padded = extracted_watermark_actual
        print("Извлеченный ЦВЗ уже имеет целевую форму 64x64.")

    # Если применялось преобразование Арнольда при встраивании (не реализовано здесь),
    # здесь нужно применить обратное преобразование к extracted_watermark_padded (64x64).
    # inv_arnold_wm = inverse_arnold_transform(extracted_watermark_padded, arnold_iterations)
    # return inv_arnold_wm

    # Возвращаем дополненную матрицу 64x64
    return extracted_watermark_padded


# --- Пример использования (только LR) ---
# Замените на реальные пути к вашим файлам
#image_container_path = 'C:\importantpapka\study\steganography\Interblock_Difference\pic\pic2.png'
image_container_path = r'C:\importantpapka\study\steganography\Interblock_Difference\pic\Screenshot_2.png'
watermark_logo_path = r'C:\importantpapka\study\steganography\Interblock_Difference\pic\watermark.png' # Используйте 64x64 ЦВЗ

# # Встраивание (только LR)
watermarked_image = embed_watermark_blue_channel_lr(
     image_container_path,
     watermark_logo_path,
     T, K, Z, COEFF_COORDS, BLOCK_SIZE
 )

if watermarked_image is not None:
     cv2.imwrite('watermarked_image_blue_channel_lr.png', watermarked_image)
     print("Изображение с водяным знаком сохранено как 'watermarked_image_blue_channel_lr.png'")

# Извлечение (только LR)
extracted_wm = extract_watermark_blue_channel_lr_64x64_dominant_pad_fixed(
     'watermarked_image_blue_channel_lr.png', # Путь к сохраненному изображению с ЦВЗ
     T, 12, COEFF_COORDS, BLOCK_SIZE
 )

inverted_extracted_wm = 1 - extracted_wm


if extracted_wm is not None:
     print("\nИзвлеченный водяной знак:")
     print(extracted_wm)
     # Извлеченный ЦВЗ теперь будет иметь форму (64, 63) для LR
     if extracted_wm.ndim == 2:
        if extracted_wm.shape[0] > 0 and extracted_wm.shape[1] > 0:
              cv2.imwrite('extracted_watermark_inverted_colors_fixed.png', inverted_extracted_wm.astype(np.uint8) * 255)
              print(f"Извлеченный ЦВЗ сохранен как 'extracted_watermark_blue_channel_lr_result.png' с формой {extracted_wm.shape}")
        else:
              print("Извлеченный ЦВЗ пустой или имеет нулевой размер, не может быть сохранен как изображение.")
     else:
         print("Извлеченный ЦВЗ не является 2D массивом и не может быть сохранен как изображение.")