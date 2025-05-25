from PIL import Image
import numpy as np
from main import arnold_transform 
from main import inverse_arnold_transform

path = 'C:\importantpapka\study\steganography\Interblock Difference\pic\watermark.png'

img = Image.open(path)
binary_img = img.convert('1')
binary_img = np.array(binary_img)

arnold = arnold_transform(binary_img, 1)
Image.fromarray(arnold).show()
iarnold = inverse_arnold_transform(arnold, 1)
Image.fromarray(iarnold).show()