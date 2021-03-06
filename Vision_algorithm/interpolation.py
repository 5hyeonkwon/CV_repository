# -*- coding: utf-8 -*-
"""중간과사과제_02_20162497

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tcF_1ERkvZjkPBh92S0E7b1eJFtbTsOZ

중간고사 과제 2

- 주어진 영상에 대하여 아래 순서에 따라 다양한 보간법에 의하여 2배 큰 영상을 구하시오
"""

# Commented out IPython magic to ensure Python compatibility.
import cv2
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

def plot_img(images, titles):
  fig, axs = plt.subplots(nrows = 1, ncols = len(images), figsize = (15, 15))
  for i, p in enumerate(images):
    axs[i].imshow(p, 'gray')
    axs[i].set_title(titles[i])
    #axs[i].axis('off')
  plt.show()

input_img = cv2.imread('test.jpeg')


gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img, 'gray')
plt.show()

"""이미지 resize(크기조정)"""

scale_percent = 50 # percent of original size
width = int(gray_img.shape[1] * scale_percent / 100)
height = int(gray_img.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
img = cv2.resize(gray_img, dim, interpolation = cv2.INTER_AREA)

plt.imshow(img, 'gray')
plt.show()

img.shape


#Neareest Neighbor Interpotation 함수를 작성하시오
def nearest_neighbor_interpolation(img, scale_factor) :
  h,w = img.shape
  answer = np.zeros(shape = (h* scale_factor, w * scale_factor))
  for i in range(h*scale_factor) :
    origin_i = i // scale_factor
    for j in range(w*scale_factor) :
      origin_j = j // scale_factor
      answer[i][j] = img[origin_i][origin_j]

  return answer



#2배 확대하였을 때 영상을 가시화

nearest_2 = nearest_neighbor_interpolation(img, 2)
images = [img, nearest_2]
titles = ['Original image', 'Nearest Neighbor Interporlation - 2times']
plot_img(images, titles)


#4배 확대하였을 때 영상을 가시화

nearest_4 = nearest_neighbor_interpolation(img, 4)
images = [img, nearest_4]
titles = ['Original image', 'Nearest Neighbor Interporlation - 4times']
plot_img(images, titles)



#Bilnear Interpotation 함수를 작성
def bilinear_interpolation(img, scale_factor) : 
  h,w = img.shape
  answer = np.zeros(shape = (int(h*scale_factor), int(w * scale_factor)))

  for i in range(answer.shape[0]) :
    for j in range(answer.shape[1]) :
      origin_height = i / scale_factor  
      origin_width = j / scale_factor # 

      y = int(i / scale_factor)
      x = int(j / scale_factor)

      l, r, t, b = x, min(x+1,w-1), y, min(y+1, h-1) # overflow 방지
      
      beta = origin_height - y
      alpha = 1 - beta
      p = origin_width - x
      q = 1 - p

      A = img[b, l]   # 위의 그림에서 A 좌표 값
      B = img[t, l]   # 위의 그림에서 B 좌표 값
      C = img[t, r]   # 위 그림에서 C 좌표 값
      D = img[b, r]   # 위 그림에서 D 좌표 값

      M = beta * A + alpha * B  # y left에서 interpolation 
      N = beta * D + alpha * C # y right에서 interpolation
      p_coor = q * beta * A + q * alpha * B + p*beta * D + p*alpha * C # 주어진 공식 연산, M, N을 x에 대하여 interpolation

      answer[i][j] = np.round(p_coor, 1) 

  return answer



#2배 확대하였을 때 영상을 가시화

bilinear_2 = bilinear_interpolation(img, 2)
images = [img, bilinear_2]
titles = ['Original image', 'Bilinear Interporlation - 2times']
plot_img(images, titles)

#4배 확대하였을 때 영상을 가시화
bilinear_4 = bilinear_interpolation(img, 4)
images = [img, bilinear_4]
titles = ['Original image', 'Bilinear Interporlation - 4times']
plot_img(images, titles)