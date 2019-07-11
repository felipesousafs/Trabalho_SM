import numpy as np
import cv2
from matplotlib import pyplot as plt

img_grey = cv2.imread('dados.jpeg', 0) # IMG original, com 5 dados.
# img_grey = cv2.imread('dados_alterados.jpeg', 0) # IMG recortada pra testar com 3 dados.
img = cv2.blur(img_grey, (20, 20))
# cv2.imshow('image', img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


titles = ['No Blur', 'With BLUR', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img_grey, img, thresh1, thresh2]

for i in range(4):
    plt.subplot(1, 4, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

print('Quantidade de dados: ')
print(len(contours))

plt.show()
