import cv2
import numpy as np

# Чтение изображения
image = cv2.imread('img.jpg')

# Изменение размера изображения
width = 400
height = 600
resized_image = cv2.resize(image, (width, height))

# Перевод в различные цветовые пространства (например, из BGR в оттенки серого)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Арифметическая операция - увеличение яркости изображения
brightened_image = cv2.add(image, np.array([50.0]))

# Комбинация операций
# Пример: увеличение яркости и перевод изображения в оттенки серого
combined_operations_image = cv2.cvtColor(cv2.add(image, np.array([50.0])), cv2.COLOR_BGR2GRAY)

# Отображение изображений
cv2.imshow('Original Image', image)
cv2.imshow('Resized Image', resized_image)
cv2.imshow('Gray Image', gray_image)
cv2.imshow('Brightened Image', brightened_image)
cv2.imshow('Combined Operations Image', combined_operations_image)

# Ожидание нажатия клавиши и закрытие окон
cv2.waitKey(0)
cv2.destroyAllWindows()