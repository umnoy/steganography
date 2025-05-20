from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_image(path):
    try:
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except Exception as e:
        print(f"Ошибка при загрузке изображения: {e}")
        return None


def plot_histograms(arr, title_prefix):
    colors = ('red', 'green', 'blue')
    plt.figure(figsize=(12, 4))
    for i, color in enumerate(colors):
        plt.subplot(1, 3, i + 1)
        plt.hist(arr[:, :, i].flatten(), bins=256, color=color, alpha=0.7)
        plt.title(f'{title_prefix} - {color.upper()} канал')
        plt.xlabel("Яркость (0-255)")
        plt.ylabel("Частота")
        plt.grid(True)
    plt.tight_layout()
    plt.show()

path1 = "C:\importantpapka\study\steganography\embedded.png"
path2 = "C:\importantpapka\study\steganography\interpolated.png"

img1 = load_image(path1)
img2 = load_image(path2)

if img1 is not None:
    plot_histograms(img1, "Изображение 1")

if img2 is not None:
    plot_histograms(img2, "Изображение 2")
