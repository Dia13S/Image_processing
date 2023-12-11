import cv2
import numpy as np

def canny_edge_detection(frame): 
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
    
    lower_limit_y  = np.array([0, 150, 210])
    upper_limit_y = np.array([40 ,200, 240])
    lower_limit_g  = np.array([45,100, 80])
    upper_limit_g = np.array([70, 200, 120])

    mask_y = cv2.inRange(frame_hsv,lower_limit_y, upper_limit_y)
    mask_g = cv2.inRange(frame_hsv,lower_limit_g, upper_limit_g)
    mask = mask_y + mask_g
    image1 = cv2.bitwise_and(frame, frame, mask=mask)

    blur = cv2.GaussianBlur(src=image1, ksize=(7, 7), sigmaX=0.5)  
    frame_canny = cv2.Canny(blur, 100, 200) 
    return blur, frame_canny

cap = cv2.VideoCapture(0) 
while True: 
    ret, frame = cap.read() 
    if not ret: 
        print('Image not captured') 
        break
           
    blur, frame_canny = canny_edge_detection(frame) 
    cv2.imshow("Размытое изображение", blur) 
    cv2.imshow("Выделение границ", frame_canny) 
          
    if cv2.waitKey(1) & 0xFF == 27: 
        break
      
cap.release() 
cv2.destroyAllWindows()