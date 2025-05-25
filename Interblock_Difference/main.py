import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn
import math 


def apply_dct(block):
    """Применяет прямое ДКП к блоку 8x8."""
    return dctn(block, type=2, norm='ortho')

def apply_idct(dct_coeffs):
    """Применяет обратное ДКП к блоку коэффициентов 8x8."""
    return idctn(dct_coeffs, type=2, norm='ortho')

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

# --- Параметры алгоритма ---
T_PARAM = 80
K_PARAM = 12
Z_PARAM = 2 # Параметр Z для M(Bp,Q)
ARNOLD_ITERATIONS = 10 # Количество итераций Арнольда для перемешивания ЦВЗ
# Выбранный среднечастотный коэффициент в блоке 8x8 (например, строка 4, столбец 5)
# Нужно выбрать один из среднечастотных диапазонов на рисунке 1.
# Координаты (строка, столбец), отсчет с 0:
EMBEDDING_COORD = (3, 4) # Например, B(4,5) на рисунке 1

# Метод выбора соседнего блока (LR, UD, RL, DU)
# Для простоты возьмем LR (слева направо)
NEIGHBOR_METHOD = 'LR'

# --- Вспомогательные функции ---

def get_median_of_ac_coeffs(dct_block):
    """
    Вычисляет медиану первых 9 АС-коэффициентов ДКП-блока.
    Исключаем DC-коэффициент (0,0). Первые 9 AC - это (0,1), (0,2), ..., (2,2) по зигзагу,
    или можно просто взять все, кроме (0,0) и отсортировать первые 9.
    Проще взять первые 9, исключив (0,0).
    """
    # Плоский список всех коэффициентов, кроме DC
    ac_coeffs = [dct_block[i, j] for i in range(8) for j in range(8) if not (i == 0 and j == 0)]
    # Возьмем первые 9 AC по плоскому списку (зигзаг будет другой порядок,
    # но для медианы это не принципиально, главное - количество)
    first_9_ac = ac_coeffs[:9]
    return np.median(first_9_ac)

def calculate_modification_strength(dct_block, Z):
    """
    Вычисляет силу модификации M(Bp,Q) по формуле из задания.
    """
    dc_coeff = dct_block[0, 0]
    median_ac = get_median_of_ac_coeffs(dct_block)

    if abs(dc_coeff) > 1000 or abs(dc_coeff) < 1:
        # Используем первую часть формулы
        M = abs(Z * median_ac)
    else:
        # Используем вторую часть формулы
        M = abs(Z * (dc_coeff - median_ac) / dc_coeff)

    # Согласно формулам встраивания, M прибавляется или вычитается.
    # Убедимся, что M неотрицательно, как в формуле M(Bp,Q)
    return max(1.0, M) # Добавим минимальное значение для стабильности

def get_neighbor_block_coords(p, q, image_shape, block_size=8, method='LR'):
    """
    Возвращает координаты (p_neighbor, q_neighbor) соседнего блока
    для блока с координатами (p, q) по выбранному методу.
    p, q - индексы блока по вертикали и горизонтали.
    image_shape - форма исходного изображения (высота, ширина).
    """
    num_blocks_v = image_shape[0] // block_size
    num_blocks_h = image_shape[1] // block_size

    if method == 'LR': # Слева направо
        if q < num_blocks_h - 1:
            return p, q + 1
        else:
            # Если это последний блок в строке, переходим на начало следующей строки (или последний блок)
            return p + 1 if p < num_blocks_v - 1 else p, 0 if p < num_blocks_v - 1 else q # Простая цикличность или последний
    # TODO: Реализовать другие методы: UD, RL, DU

    # Для примера LR: если блок последний в строке, взять первый блок в следующей.
    # Это может отличаться от логики в статье, там возможно просто берется последний возможный блок.
    # Упрощенный вариант LR (только вправо, если возможно):
    if method == 'LR':
         if q < num_blocks_h - 1:
             return p, q + 1
         else:
             # Если справа нет, то берем текущий? Или ошибка?
             # Предположим, что в примере из статьи идет последовательный обход блоков.
             # Если мы дошли до конца строки, берем следующий блок в следующей строке.
             if p < num_blocks_v - 1:
                 return p + 1, 0
             else:
                 # Если это последний блок изображения, соседа нет.
                 # В реальной реализации нужно подумать, как обрабатывать границы.
                 # Возможно, просто не встраивать в блоки, у которых нет соседа по выбранному направлению.
                 # Для этого шаблона вернем None, чтобы показать, что нет соседа.
                 return None, None

    # Реализуй другие методы по аналогии или согласно уточнению в работе
    # if method == 'UD': ...
    # if method == 'RL': ...
    # if method == 'DU': ...

    return None, None # На случай, если метод не реализован или нет соседа

# --- Основные функции алгоритма ---

def embed_watermark(image_path, watermark_path, output_path):
    """
    Встраивает бинарный водяной знак в частотную область изображения-контейнера.

    Args:
        image_path (str): Путь к изображению-контейнеру.
        watermark_path (str): Путь к изобранарному водяному знаку.
        output_path (str): Путь для сохранения изображения с ЦВЗ.
    """
    # 1. Загрузка изображений
    try:
        img = Image.open(image_path).convert('L') # Переводим в оттенки серого
        watermark_img = Image.open(watermark_path).convert('L') # Водяной знак тоже в оттенках серого
    except FileNotFoundError:
        print("Ошибка: Файл изображения или водяного знака не найден.")
        return

    img_array = np.array(img, dtype=np.float64) # Используем float64 для точности ДКП
    watermark_array = np.array(watermark_img, dtype=np.float64)

    # Убедимся, что водяной знак бинарный (0 или 255, переведем в 0 или 1)
    watermark_array = (watermark_array > 128).astype(int) # Считаем, что > 128 это 1, иначе 0

    # Проверка размеров
    img_height, img_width = img_array.shape
    wm_height, wm_width = watermark_array.shape

    if img_height != img_width:
        print("Ошибка: Изображение-контейнер должно быть квадратным.")
        return
    if wm_height != wm_width:
         print("Ошибка: Водяной знак должен быть квадратным.")
         return
    if wm_height >= img_height or wm_width >= img_width:
        print("Ошибка: Размер водяного знака должен быть меньше размера контейнера.")
        return
    if img_height % 8 != 0 or img_width % 8 != 0:
         print("Предупреждение: Размер изображения-контейнера не кратен 8. Могут быть неполные блоки.")
         # В реальной реализации нужно обрезать или дополнить изображение

    block_size = 8
    num_blocks_v = img_height // block_size
    num_blocks_h = img_width // block_size

    if wm_height * 8 > img_height or wm_width * 8 > img_width:
         print("Ошибка: Размер водяного знака слишком велик для встраивания по одному биту на блок.")
         return

    # 2. Предобработка контейнера: сдвиг значений
    img_array_shifted = img_array - 128

    # 3. Предобработка водяного знака: преобразование Арнольда
    scrambled_watermark = arnold_transform(watermark_array, ARNOLD_ITERATIONS)
    print(f"Водяной знак перемешан ({ARNOLD_ITERATIONS} итераций Арнольда).")

    # Создаем копию массива для модификации
    modified_img_array_shifted = np.copy(img_array_shifted)

    # 4. Встраивание битов ЦВЗ
    watermark_bit_index = 0
    total_watermark_bits = wm_height * wm_width
    bits_embedded_count = 0

    print("Начинаем встраивание...")

    # Итерируем по блокам изображения-контейнера
    for p in range(num_blocks_v):
        for q in range(num_blocks_h):

            if watermark_bit_index >= total_watermark_bits:
                break # Все биты водяного знака встроены

            # Извлекаем текущий блок
            block = modified_img_array_shifted[p*block_size:(p+1)*block_size, q*block_size:(q+1)*block_size]

            # Применяем ДКП
            dct_block = apply_dct(block)

            # Находим соседний блок
            p_neighbor, q_neighbor = get_neighbor_block_coords(p, q, img_array.shape, block_size, NEIGHBOR_METHOD)

            if p_neighbor is None or q_neighbor is None:
                 # Пропускаем блоки без соседа по выбранному методу (например, крайние)
                 #print(f"Пропускаем блок ({p},{q}) - нет соседа.")
                 continue

            # Извлекаем соседний блок (для расчета разности)
            neighbor_block = modified_img_array_shifted[p_neighbor*block_size:(p_neighbor+1)*block_size, q_neighbor*block_size:(q_neighbor+1)*block_size]
            dct_neighbor_block = apply_dct(neighbor_block)

            # Выбираем коэффициенты для встраивания и соседа
            coeff_p_q = dct_block[EMBEDDING_COORD]
            coeff_neighbor = dct_neighbor_block[EMBEDDING_COORD]

            # Рассчитываем исходную разность
            delta = coeff_p_q - coeff_neighbor

            # Рассчитываем силу модификации M(Bp,Q) для текущего блока Bp,Q
            M = calculate_modification_strength(dct_block, Z_PARAM)

            # Получаем текущий бит водяного знака
            wm_bit_row = watermark_bit_index // wm_width
            wm_bit_col = watermark_bit_index % wm_width
            current_watermark_bit = scrambled_watermark[wm_bit_row, wm_bit_col]

            # --- Логика встраивания бита (согласно формулам из задания) ---
            # ЭТО МЕСТО НУЖНО ЗАПОЛНИТЬ СОГЛАСНО ПРАВИЛАМ ДЛЯ БИТА 0 и БИТА 1
            # Пример структуры для бита 1:
            # while True:
            #     delta = coeff_p_q - coeff_neighbor # Пересчитываем дельту
            #     modified_coeff_p_q = coeff_p_q # Начальное значение
            #     condition_met = False
            #
            #     if current_watermark_bit == 1:
            #         # Вариант 1 для бита 1: Delta > T - K
            #         if delta > T_PARAM - K_PARAM:
            #             # Условие выполнено, возможно, нужно скорректировать, если оно перестало выполняться
            #             # Но по заданию, пока Delta > T - K, нужно выполнять корректировку
            #             # modified_coeff_p_q = coeff_p_q - M # Пример корректировки
            #             # condition_met = True # Условие выполнено
            #             pass # Логика корректировки для случая 1 бита 1
            #
            #         # Вариант 2 для бита 1: K > Delta > -T/2
            #         elif K_PARAM > delta > -T_PARAM / 2:
            #             # modified_coeff_p_q = coeff_p_q + M # Пример корректировки
            #             # condition_met = True
            #             pass # Логика корректировки для случая 2 бита 1
            #
            #         # Вариант 3 для бита 1: Delta < -T/2
            #         elif delta < -T_PARAM / 2:
            #              # modified_coeff_p_q = coeff_p_q - M # Пример корректировки
            #              # condition_met = True
            #              pass # Логика корректировки для случая 3 бита 1
            #
            #     elif current_watermark_bit == 0:
            #         # Логика для бита 0 (аналогично, три варианта)
            #         # Вариант 1 для бита 0: Delta > T/2
            #         # while Delta <= T + K: ...
            #         pass # Заполни логику для бита 0
            #
            #     # Проверка, нужно ли продолжать итерации
            #     # if condition_met and НОВОЕ ЗНАЧЕНИЕ delta УДОВЛЕТВОРЯЕТ УСЛОВИЮ НУЖНОГО ИНТЕРВАЛА:
            #     #     break # Выходим из цикла, бит успешно встроен
            #     # else:
            #     #     coeff_p_q = modified_coeff_p_q # Обновляем коэффициент для следующей итерации
            #     #     # Здесь нужно добавить ограничение на количество итераций, чтобы избежать бесконечного цикла

            # Placeholder: В этом месте должна быть логика модификации coeff_p_q
            # Пример: Если бит = 1 и разность не в нужном интервале, корректируем coeff_p_q
            # Этот цикл while True с условиями выхода и корректировкой коэффициента - ключевая часть алгоритма.

            # --- Конец логики встраивания бита ---

            # ВАЖНО: После того, как ты определил final_coeff_p_q после итераций:
            # Обнови коэффициент в ДКП-блоке:
            # dct_block[EMBEDDING_COORD] = final_coeff_p_q

            # Применяем обратное ДКП к МОДИФИЦИРОВАННОМУ блоку ДКП
            restored_block_shifted = apply_idct(dct_block)

            # Обновляем соответствующий блок в массиве изображения
            modified_img_array_shifted[p*block_size:(p+1)*block_size, q*block_size:(q+1)*block_size] = restored_block_shifted

            watermark_bit_index += 1
            bits_embedded_count += 1

    print(f"Встраивание завершено. Встроено {bits_embedded_count} битов.")


    # 5. Постобработка контейнера: сдвиг значений обратно и округление/обрезка
    final_img_array = modified_img_array_shifted + 128
    final_img_array = np.round(final_img_array) # Округляем до целых
    final_img_array = np.clip(final_img_array, 0, 255) # Обрезаем до диапазона [0, 255]
    final_img_array = final_img_array.astype(np.uint8) # Приводим к типу uint8 для Pillow

    # 6. Сохранение изображения
    output_img = Image.fromarray(final_img_array, mode='L') # 'L' для оттенков серого
    output_img.save(output_path)
    print(f"Изображение с ЦВЗ сохранено как {output_path}")


def extract_watermark(image_path, original_image_shape, watermark_shape, output_path):
    """
    Извлекает бинарный водяной знак из изображения.

    Args:
        image_path (str): Путь к изображению, содержащему ЦВЗ.
        original_image_shape (tuple): Форма (высота, ширина) оригинального изображения-контейнера.
                                      Нужна для правильной итерации по блокам.
        watermark_shape (tuple): Форма (высота, ширина) оригинального водяного знака.
                                   Нужна для восстановления формы и обратного Арнольда.
        output_path (str): Путь для сохранения извлеченного бинарного водяного знака.
    """
    # 1. Загрузка изображения с ЦВЗ
    try:
        img_with_wm = Image.open(image_path).convert('L') # В оттенках серого
    except FileNotFoundError:
        print("Ошибка: Файл изображения не найден.")
        return

    img_array_with_wm = np.array(img_with_wm, dtype=np.float64)

    # Проверка размеров
    img_height, img_width = img_array_with_wm.shape
    orig_img_height, orig_img_width = original_image_shape
    wm_height, wm_width = watermark_shape

    if img_height != orig_img_height or img_width != orig_img_width:
        print("Предупреждение: Размеры изображения с ЦВЗ отличаются от размеров оригинального контейнера.")
        # Это может быть следствием атаки (масштабирование, кадрирование)
        # В робастном извлечении нужно обрабатывать такие случаи.
        # Для базового извлечения в отсутствие атак, просто используем текущие размеры,
        # но нужно понимать, что для робастности это недостаточно.
        print("Используем текущие размеры изображения для извлечения.")
        # В реальной робастной системе нужно сделать пре-обработку для восстановления исходных размеров/пропорций.


    block_size = 8
    # Используем размеры изображения с ЦВЗ для итерации по блокам
    num_blocks_v = img_height // block_size
    num_blocks_h = img_width // block_size

    # 2. Предобработка: сдвиг значений
    img_array_with_wm_shifted = img_array_with_wm - 128

    extracted_bits = []

    print("Начинаем извлечение...")

    # 3. Извлечение битов ЦВЗ
    watermark_bit_count = 0
    # В идеале, мы должны извлечь ровно столько битов, сколько было в исходном ЦВЗ
    total_watermark_bits_expected = wm_height * wm_width

    # Итерируем по блокам изображения
    for p in range(num_blocks_v):
        for q in range(num_blocks_h):

            if watermark_bit_count >= total_watermark_bits_expected:
                break # Достаточно битов извлечено

            # Извлекаем текущий блок
            block = img_array_with_wm_shifted[p*block_size:(p+1)*block_size, q*block_size:(q+1)*block_size]

            # Применяем ДКП
            dct_block = apply_dct(block)

            # Находим соседний блок (тем же методом, что при встраивании)
            p_neighbor, q_neighbor = get_neighbor_block_coords(p, q, img_array_with_wm.shape, block_size, NEIGHBOR_METHOD)

            if p_neighbor is None or q_neighbor is None:
                # Пропускаем блоки, если у них нет соседа (как при встраивании)
                continue

            # Извлекаем соседний блок
            neighbor_block = img_array_with_wm_shifted[p_neighbor*block_size:(p_neighbor+1)*block_size, q_neighbor*block_size:(q_neighbor+1)*block_size]
            dct_neighbor_block = apply_dct(neighbor_block)

            # Выбираем те же коэффициенты, что использовались при встраивании
            coeff_p_q = dct_block[EMBEDDING_COORD]
            coeff_neighbor = dct_neighbor_block[EMBEDDING_COORD]

            # Рассчитываем разность
            delta = coeff_p_q - coeff_neighbor

            # --- Логика извлечения бита (согласно формулам из задания) ---
            # W_i = {1, если (Δ < -T) или ((Δ > 0) и (Δ < T)),
            #        0, если (Δ > T) или ((Δ < 0) и (Δ > -T)).

            extracted_bit = -1 # -1 для индикации ошибки или неопределенности

            if (delta < -T_PARAM) or ((delta > 0) and (delta < T_PARAM)):
                 extracted_bit = 1
            elif (delta > T_PARAM) or ((delta < 0) and (delta > -T_PARAM)):
                 extracted_bit = 0
            # Внимание: есть промежутки (-T, -T/2] и [K, T], а также [-T/2, 0]
            # и [0, K) которые явно не описаны в правилах извлечения.
            # Предполагаем, что при корректном встраивании разность попадает в один из описанных интервалов.
            # Возможно, есть опечатка в правилах извлечения или требуется дополнительная логика.
            # Согласно правилам встраивания, интервалы для бита 1: Δ > T - K, K > Δ > -T/2, Δ < -T/2
            # Интервалы для бита 0: Δ > T/2, -K < Δ < T/2, Δ < K - T

            # Сопоставим правила встраивания и извлечения, возможно правила извлечения упрощены
            # Попробуем сопоставить извлечение с интервалами встраивания для ясности:
            # Если встраивался 1: Δ > T-K, или K > Δ > -T/2, или Δ < -T/2
            # Если встраивался 0: Δ > T/2, или -K < Δ < T/2, или Δ < K-T

            # Упрощенная логика извлечения на основе интервалов из задания:
            if (delta < -T_PARAM) or (0 < delta < T_PARAM):
                 extracted_bit = 1
            elif (delta > T_PARAM) or (-T_PARAM < delta < 0): # Исправил интервал для 0
                 extracted_bit = 0
            # Иначе (-T <= delta <= 0) или (delta = T) - неопределенность?
            # Для простоты в этом шаблоне будем считать, что такие случаи не возникают при успешном встраивании без атак.

            if extracted_bit != -1:
                 extracted_bits.append(extracted_bit)
                 watermark_bit_count += 1

    print(f"Извлечение завершено. Извлечено {watermark_bit_count} битов.")

    # Проверка, достаточно ли битов извлечено
    if len(extracted_bits) < total_watermark_bits_expected:
        print(f"Внимание: Извлечено недостаточно битов ({len(extracted_bits)} из {total_watermark_bits_expected}).")
        # Если извлечено меньше, нужно либо дополнить нулями/случайными значениями, либо вернуть ошибку.
        # Для простоты шаблона, дополним нулями
        while len(extracted_bits) < total_watermark_bits_expected:
            extracted_bits.append(0)


    # Преобразуем список битов обратно в numpy массив нужной формы
    extracted_watermark_array_flat = np.array(extracted_bits[:total_watermark_bits_expected], dtype=int)
    extracted_watermark_array = extracted_watermark_array_flat.reshape((wm_height, wm_width))


    # 4. Обратное преобразование Арнольда
    restored_watermark_array = inverse_arnold_transform(extracted_watermark_array, ARNOLD_ITERATIONS)
    print(f"Извлеченный водяной знак де-перемешан ({ARNOLD_ITERATIONS} итераций обратного Арнольда).")


    # Преобразуем бинарный массив (0 и 1) в изображение (0 и 255)
    restored_watermark_img_array = restored_watermark_array * 255
    restored_watermark_img_array = restored_watermark_img_array.astype(np.uint8)


    # 5. Сохранение извлеченного водяного знака
    restored_watermark_img = Image.fromarray(restored_watermark_img_array, mode='L')
    restored_watermark_img.save(output_path)
    print(f"Извлеченный водяной знак сохранен как {output_path}")


# --- Пример использования ---

if __name__ == "__main__":
    # Создай или используй существующие файлы:
    # 'container.png' - твое изображение-контейнер (лучше квадратное, кратное 8)
    # 'watermark.png' - твой бинарный водяной знак (квадратный, меньше контейнера)
    container_image_file = 'container.png'
    watermark_image_file = 'watermark.png'
    watermarked_image_file = 'container_with_wm.png'
    extracted_watermark_file = 'extracted_wm.png'

    # Получим форму оригинального контейнера и водяного знака для извлечения
    try:
        orig_img = Image.open(container_image_file)
        orig_img_shape = orig_img.size[::-1] # Pillow size (width, height), numpy shape (height, width)
        wm_img = Image.open(watermark_image_file)
        wm_shape = wm_img.size[::-1]
        orig_img.close()
        wm_img.close()
    except FileNotFoundError:
        print("Не найдены файлы для примера использования. Убедитесь, что 'container.png' и 'watermark.png' существуют.")
        # Можешь создать заглушки для тестирования:
        # orig_img_shape = (512, 512)
        # wm_shape = (64, 64)
        # Создай np.zeros((512, 512), dtype=np.uint8) и Image.fromarray(...).save('container.png')
        # Создай np.random.randint(0, 2, size=(64, 64), dtype=np.uint8) * 255 и Image.fromarray(...).save('watermark.png')
        exit()


    # Шаг 1: Встроить водяной знак
    print("\n--- Встраивание ---")
    embed_watermark(container_image_file, watermark_image_file, watermarked_image_file)

    # Шаг 2: Извлечь водяной знак
    print("\n--- Извлечение ---")
    # При извлечении нужно знать оригинальные размеры контейнера и водяного знака
    extract_watermark(watermarked_image_file, orig_img_shape, wm_shape, extracted_watermark_file)

    # Шаг 3 (Опционально): Оценить незаметность и робастность (требует реализации функций оценки)
    # print("\n--- Оценка ---")
    # from metrics import calculate_psnr, calculate_ssim, calculate_nc # Пример импорта метрик
    #
    # original_img = Image.open(container_image_file).convert('L')
    # watermarked_img = Image.open(watermarked_image_file).convert('L')
    # extracted_wm_img = Image.open(extracted_watermark_file).convert('L')
    # original_wm_img = Image.open(watermark_image_file).convert('L')
    #
    # psnr_value = calculate_psnr(np.array(original_img), np.array(watermarked_img))
    # ssim_value = calculate_ssim(np.array(original_img), np.array(watermarked_img))
    # nc_value_no_attack = calculate_nc(np.array(original_wm_img) > 128, np.array(extracted_wm_img) > 128) # Сравниваем бинарные матрицы
    #
    # print(f"Незаметность:")
    # print(f"  PSNR (Original vs Watermarked): {psnr_value:.2f} dB")
    # print(f"  SSIM (Original vs Watermarked): {ssim_value:.4f}")
    # print(f"Робастность (без атак):")
    # print(f"  NC (Original WM vs Extracted WM): {nc_value_no_attack:.4f}")

    # TODO: Добавить тесты робастности с JPEG-сжатием и другими атаками.
    # Для этого нужно будет загрузить watermarked_image_file, применить к нему атаку (например, сохранить с разным качеством JPEG),
    # затем извлечь ЦВЗ из атакованного изображения и посчитать NC.