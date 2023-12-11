import cv2
import matplotlib.pyplot as plt
import numpy as np

cv2.namedWindow('Window')

def nothing(x):
    pass

cv2.createTrackbar('Image', 'Window', 0, 7, nothing)
while True:
    picture = cv2.getTrackbarPos('Image', 'Window')
    # img = np.load("img_1.npz")["frame"]
    image = cv2.imread('img_' + str(picture+ + 1) + '.jpeg')
    image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image_rgb,cv2.COLOR_RGB2HSV)

    # fig, ax = plt.subplots(1, 3)
    # ax[0].set_title('H-histogram', fontsize = 18)
    # ax[1].set_title('S-histogram', fontsize = 18)
    # ax[2].set_title('V-histogram', fontsize = 18)
    # ax[0].set_xlabel('Bin', fontsize = 13)
    # ax[1].set_xlabel('Bin', fontsize = 13)
    # ax[2].set_xlabel('Bin', fontsize = 13)
    # ax[0].set_ylabel('Frequency', fontsize = 13)

    # h, s, v = image_hsv[:,:,2], image_hsv[:,:,1], image_hsv[:,:,0]
    # hist_h = cv2.calcHist([h],[0],None,[360],[0,360])
    # hist_s = cv2.calcHist([s],[0],None,[256],[0,256])
    # hist_v = cv2.calcHist([v],[0],None,[256],[0,256])

    lower_limit_y  = np.array([100, 0, 200])
    upper_limit_y = np.array([180, 27, 280])

    mask_y = cv2.inRange(image_hsv,lower_limit_y, upper_limit_y)
    image1 = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_y)

    # Применение фильтров
    blur = cv2.GaussianBlur(image1, (7,7), 0)
    img_canny1 = cv2.Canny(blur,50,200)

    # ax[0].plot(hist_h, color='r')
    # ax[1].plot(hist_s, color='g')
    # ax[2].plot(hist_v, color='b')

    contours, hierarchies = cv2.findContours(img_canny1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]

    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cv2.drawContours(image, [cnt], -1, (0, 0, 255), 2)
    cv2.circle(image, (cx, cy), 7, (0, 0, 255), -1)
    cv2.putText(image, 'center: ' + str(cx)+' '+ str(cy) , (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
   
    
    #print(cx, cy)

    # fig1, ax1 = plt.subplots(2, 2)
    # plt.figure(1)
    # ax1[0,0].imshow(mask_y, cmap = 'gray')
    # ax1[0,1].imshow(image, cmap = 'gray')
    # ax1[1,0].imshow(img_canny1, cmap = 'gray')
    # plt.show()
    cv2.imshow('Window', image)
    key = cv2.waitKey(1)
    if key ==27:
        break

cv2.destroyAllWindows()