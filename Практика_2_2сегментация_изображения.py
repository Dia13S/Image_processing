import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('img1.png')
image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image_hsv = cv2.cvtColor(image_rgb,cv2.COLOR_RGB2HSV)
h = s = v = 0

# Построение гистограмм
fig, ax = plt.subplots(1, 3)
ax[0].set_title('H-histogram', fontsize = 18)
ax[1].set_title('S-histogram', fontsize = 18)
ax[2].set_title('V-histogram', fontsize = 18)
ax[0].set_xlabel('Bin', fontsize = 13)
ax[1].set_xlabel('Bin', fontsize = 13)
ax[2].set_xlabel('Bin', fontsize = 13)
ax[0].set_ylabel('Frequency', fontsize = 13)

h, s, v = image_hsv[:,:,2], image_hsv[:,:,1], image_hsv[:,:,0]
hist_h = cv2.calcHist([h],[0],None,[256],[0,256])
hist_s = cv2.calcHist([s],[0],None,[256],[0,256])
hist_v = cv2.calcHist([v],[0],None,[256],[0,256])

ax[0].plot(hist_h, color='r')
ax[1].plot(hist_s, color='g')
ax[2].plot(hist_v, color='b')

# Определение цвета нажатием клавиши
# def color_function(x,y,event,flags,param):
#     global h, s, v
#     if event == cv2.EVENT_LBUTTONUP:
#         h,s,v = image_hsv[y,x]
#         print(h)
#         print(s)
#         print(v)
# cv2.setMouseCallback('image', color_function)

# Пороговые значения цветов
lower_limit_y  = np.array([0, 150, 210])
upper_limit_y = np.array([40 ,200, 240])
lower_limit_g  = np.array([45, 100, 80])
upper_limit_g = np.array([70, 200, 120])

mask_y = cv2.inRange(image_hsv,lower_limit_y, upper_limit_y)
mask_g = cv2.inRange(image_hsv,lower_limit_g, upper_limit_g)
mask = mask_y + mask_g
image1 = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

# Применение фильтров
blur = cv2.GaussianBlur(image1, (7,7), 0)
blur49 = cv2.GaussianBlur(image_rgb, (49,49), 0)
img_canny1 = cv2.Canny(blur,50,200)
img_canny2 = cv2.Canny(image1,50,200)
result = cv2.subtract(image_rgb,blur49)

fig1, ax1 = plt.subplots(2, 3)
plt.figure(1)
ax1[0,0].imshow(image1)
ax1[0,1].imshow(result)
ax1[1,0].imshow(img_canny1, cmap = 'gray')
ax1[1,1].imshow(img_canny2, cmap = 'gray')
ax1[0,2].imshow(image_rgb)
ax1[0,0].set_title('Выделенные объекты')
ax1[0,1].set_title('Фильтр Гаусса')
ax1[1,0].set_title('Фильтр Кэнни с размытием')
ax1[1,1].set_title('Фильтр Кэнни без размытия')
ax1[0,2].set_title('Исходное изображение')
plt.show()

# cv2.namedWindow('image')
# while True:
#     cv2.imshow('image1', mask)
#     cv2.imshow('image', image_hsv)
#     if cv2.waitKey(20) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()