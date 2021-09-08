import cv2
import numpy as np

def canny(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #canny = cv2.Canny(blur,low_threshold,high_threshold)
    canny = cv2.Canny(blur,50,150)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
    [(200,height),(1100,height),(550,250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        print(line)
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)

canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)

lines = cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5) #Threshold optimum value is 100
average_lines = average_slope_intercept(lane_image,lines)

cv2.imshow('result',average_lines)
cv2.waitKey(0)
