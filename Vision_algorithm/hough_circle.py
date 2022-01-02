# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('coins.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
height, width = gray.shape
print(height, width)

plt.imshow(gray, cmap='gray')
plt.show()

img_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
img_sobel_x = cv2.convertScaleAbs(img_sobel_x)

img_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
img_sobel_y = cv2.convertScaleAbs(img_sobel_y)


img_sobel = cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 1, 0);

plt.figure(figsize=(14,7))
plt.subplot(131)
plt.imshow(img_sobel_x, cmap='gray')

plt.subplot(132)
plt.imshow(img_sobel_y, cmap='gray')

plt.subplot(133)
plt.imshow(img_sobel, cmap='gray')

plt.show()

threshold, binary_img = cv2.threshold(img_sobel, 245, 255, cv2.THRESH_BINARY)

print(threshold)
plt.imshow(binary_img, cmap='gray')
plt.show()


# edge를 찾기 위한 function을 따로 설정하여 시간 복잡도를 줄임 
def find_edge(count_dict, x,y,height,width, cos_thetas, sin_thetas, radius) :
  for cos,sin in zip(cos_thetas, sin_thetas) : 
    # 삼각함수 공식을 이용한 center 값 찾는 방법 음수이거나 width나 height 보다 크면 -1로 설정하여 예외처리
    center_x = x - int(cos * radius) if 0 < (x - int(cos * radius)) < width else -1 
    center_y = y - int(sin * radius) if 0 < (y - int(sin * radius)) < height else -1
    if (center_x, center_y, radius) in count_dict.keys() :
      if center_x == -1 or center_y == -1 :
        continue
      else :
        # voting 
        count_dict[(center_x, center_y, radius)] +=1 
    else :
      if center_x == -1 or center_y == -1 :
        continue
      else :
        count_dict[(center_x, center_y, radius)] = 1

def Hough_Circles(binary_img, min_r, max_r, quant_theta, quant_r ,threshold):
  count_dict = dict() # 원의 반지름과 좌표를 저장하기 위한 dictionary
  
  h,w = binary_img.shape
  min_r = min_r
  # hough transform 실습에서 가져옴
  if max_r == -1 :
    max_r = np.sqrt(h*h+w*w)
  else :
    max_r = max_r

  quant_theta = quant_theta # theta 양자화를 위한 기준
  quant_r = quant_r# radius 양자화를 위한 기준 

  theta_range = np.arange(0,360, quant_theta) # [0,360) 기준으로 양자화 하여 center를 찾음 
  d2r = np.pi/180 # numpy의 cos, sin은 radian사용하여 degree를 radian으로 바꾸기 위해 degree to radian

  radius_range = np.arange(min_r, max_r, quant_r) # 속도를 빠르게 하기 위해 전체가 아닌 양자화를 하여 구간별로 진행
  cos_thetas = np.cos(theta_range * d2r) 
  sin_thetas = np.sin(theta_range * d2r)

  for i in range(0, h) : 
    for j in range(0, w) :
      if binary_img[i][j] == 0 :
        continue
      else :
        for radius in radius_range : #radius 기준으로 edge를 찾아 count_dict에 추가 
          find_edge(count_dict,x=j, y= i,height= h, width=w, cos_thetas= cos_thetas, sin_thetas=sin_thetas, radius=radius)



  sorted_dict = sorted([items for items in count_dict.items() if items[1] > threshold], key=lambda x: -x[1]) #threshold를 설정하여 메모리 감소
  del count_dict

  return sorted_dict

def display_Circles(binary_img, circle_coordinate): 
# 원의 중심과 원의 반지름을 이용하여 cv2.circle을 통해 원을 그린다.
  color_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
  for (x,y,r), _ in circle_coordinate:
    color_img = cv2.circle(color_img, (x,y), r, (0,255,0), 4) # green color

  plt.figure(figsize=(12,11))
  plt.title("Hough Circle Result")
  plt.imshow(color_img)
  plt.show()


circle_coordinate = Hough_Circles(binary_img = binary_img, min_r = 40, max_r= 100, quant_theta=5, quant_r=5, threshold=47) # hyperParameters

display_Circles(binary_img, circle_coordinate)