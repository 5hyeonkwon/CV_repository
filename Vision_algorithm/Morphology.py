# -*- coding: utf-8 -*-
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

"""주어진 영상을 읽고 가시화"""

img = cv2.imread('corporate-espionage.jpg', 0)
plt.imshow(img, 'gray')
plt.show()
height, width  = img.shape
print(width, height)

"""영상 이진화"""

ret, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

# Plot the images
images = [img, binary_img]
titles = ['Original image', 'THRESH_BINARY']
plot_img(images, titles)


#3x3 팽창(dilatioin) 함수 작성
def dilation_3x3(img): # binary image 입력
  re_img = np.abs(img - np.max(img)) # 검은색 부분 1 흰색 부분 0으로 바꿔줌
  filter = np.array([[0,1,0],[1,1,1],[0,1,0]]) # filter 
  filter_h, filter_w = filter.shape
  re_img = np.pad(re_img, (filter_h//2,filter_w//2), mode = 'constant', constant_values = 0) # 모서리 부분도 고려
  h, w = re_img.shape

  anchor = filter[filter_h//2, filter_w//2] # 가운데 값 
  out = np.zeros_like(re_img)
  for i in range(filter_h//2, h - filter_h//2 + 1) :
    for j in range(filter_w//2, w- filter_w//2 + 1) :
      if anchor == re_img[i][j] :
        h_diff = filter_h//2
        w_diff = filter_w//2      
        h_ = 0
        w_ = 0
        # 가운데 값이 동일하다면 filter의 4연결성을 고려하여 anchor를 제외한 값들을 1로 바꾸어줌 
        # 따라서 이전 anchor에 의해 다음 anchor가 1인지 아닌지 평가 받음
        for i_ in range(i-h_diff, i+h_diff+1) :  #1,4
          for j_ in range(j-w_diff,j+w_diff+1) : #3, 6
            if filter[h_, w_] == 1 :
              if (h_ == filter_h // 2) and (w_ == filter_w // 2) :
                w_ += 1
                continue
              else :
                out[i_,j_] = 1 
            w_ += 1
          h_ += 1
          w_ = 0




  output_img = np.abs(out - np.max(out))
  output_img = output_img[filter_h//2:h-filter_h//2, filter_w//2:w-filter_w//2]
  return output_img


#3x3 침식(erosion) 함수 작성
def erosion_3x3(img):
  re_img = np.abs(img - np.max(img)) # 검은색 부분 1 흰색 부분 0으로 바꿔줌
  filter = np.array([[0,1,0],[1,1,1],[0,1,0]]) # filter 
  filter_h, filter_w = filter.shape
  re_img = np.pad(re_img, (filter_h//2,filter_w//2), mode = 'constant', constant_values = 0) # 모서리 부분도 고려
  h, w = re_img.shape

  anchor = filter[filter_h//2, filter_w//2] # 가운데 값 

  out = np.zeros_like(re_img)
  for i in range(filter_h//2, h - filter_h//2+1) :
    for j in range(filter_w//2, w- filter_w//2 + 1) :
      if anchor == re_img[i][j] :
        h_diff = filter_h//2
        w_diff = filter_w//2      
        h_ = 0
        w_ = 0
        count = 0
        for i_ in range(i-h_diff, i+h_diff+1) :  
          for j_ in range(j-w_diff,j+w_diff+1) :
            if filter[h_, w_] == 1  and re_img[i_,j_] == 1:
              count += 1 
            w_ += 1
          h_ += 1
          w_ = 0
        if count == np.sum(filter) : 
          out[i][j] = 1

  output_img = np.abs(out - np.max(out)) # 검은색부분 0 밝은 부분 255로 바꿔줌  
  output_img = output_img[filter_h//2:h-filter_h//2, filter_w//2:w-filter_w//2] # padding 제거
  return output_img

#3x3 열기(open) 함수 작성
def open_3x3(img):
  erosion_img = erosion_3x3(img) # erosion
  output_img = dilation_3x3(erosion_img) # dilation
  
  
  return output_img

#3x3 닫기(close) 함수 작성
def close_3x3(img):
  dilated_img = dilation_3x3(img) #dilation
  output_img = erosion_3x3(dilated_img)  #erosion
  
  return output_img

#5x5 팽창(dilatioin) 함수 작성
def dilation_5x5(img): # binary image 입력
  re_img = np.abs(img - np.max(img)) # 검은색 부분 1 흰색 부분 0으로 바꿔줌
  filter = np.array([[0,0,1,0,0],[0,0,1,0,0],[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0]]) # filter 
  filter_h, filter_w = filter.shape
  re_img = np.pad(re_img, (filter_h//2,filter_w//2), mode = 'constant', constant_values = 0) # 모서리 부분도 고려
  h, w = re_img.shape

  anchor = filter[filter_h//2, filter_w//2] # 가운데 값 
  out = np.zeros_like(re_img)
  for i in range(filter_h//2, h - filter_h//2 + 1) :
    for j in range(filter_w//2, w- filter_w//2 + 1) :
      if anchor == re_img[i][j] :
        h_diff = filter_h//2
        w_diff = filter_w//2      
        h_ = 0
        w_ = 0
        # 가운데 값이 동일하다면 filter의 4연결성을 고려하여 anchor를 제외한 값들을 1로 바꾸어줌 
        # 따라서 이전 anchor에 의해 다음 anchor가 1인지 아닌지 평가 받음
        for i_ in range(i-h_diff, i+h_diff+1) :  #1,4
          for j_ in range(j-w_diff,j+w_diff+1) : #3, 6
            if filter[h_, w_] == 1 :
              if (h_ == filter_h // 2) and (w_ == filter_w // 2) :
                w_ += 1
                continue
              else :
                out[i_,j_] = 1 
            w_ += 1
          h_ += 1
          w_ = 0




  output_img = np.abs(out - np.max(out))
  output_img = output_img[filter_h//2:h-filter_h//2, filter_w//2:w-filter_w//2]
  return output_img


#5x5 침식(erosion) 함수 작성
def erosion_5x5(img):
  re_img = np.abs(img - np.max(img)) # 검은색 부분 1 흰색 부분 0으로 바꿔줌
  filter = np.array([[0,0,1,0,0],[0,0,1,0,0],[1,1,1,1,1],[0,0,1,0,0],[0,0,1,0,0]]) # filter 
  filter_h, filter_w = filter.shape
  re_img = np.pad(re_img, (filter_h//2,filter_w//2), mode = 'constant', constant_values = 0) # 모서리 부분도 고려
  h, w = re_img.shape

  anchor = filter[filter_h//2, filter_w//2] # 가운데 값 

  out = np.zeros_like(re_img)
  for i in range(filter_h//2, h - filter_h//2+1) :
    for j in range(filter_w//2, w- filter_w//2 + 1) :
      if anchor == re_img[i][j] :
        h_diff = filter_h//2
        w_diff = filter_w//2      
        h_ = 0
        w_ = 0
        count = 0
        for i_ in range(i-h_diff, i+h_diff+1) :  
          for j_ in range(j-w_diff,j+w_diff+1) :
            if filter[h_, w_] == 1  and re_img[i_,j_] == 1:
              count += 1 
            w_ += 1
          h_ += 1
          w_ = 0
        if count == np.sum(filter) : 
          out[i][j] = 1

  output_img = np.abs(out - np.max(out)) # 검은색부분 0 밝은 부분 255로 바꿔줌  
  output_img = output_img[filter_h//2:h-filter_h//2, filter_w//2:w-filter_w//2] # padding 제거
  return output_img

#5x5 열기(open) 함수 작성
def open_5x5(img):
  erosion_img = erosion_5x5(img)
  output_img = dilation_5x5(erosion_img)
  
  
  return output_img

#5x5 닫기(close) 함수 작성
def close_5x5(img):
  dilated_img = dilation_5x5(img)
  output_img = erosion_5x5(dilated_img)  
  
  return output_img

"""함수를 이용하여 가시화"""

# 팽창(dilation) 결과 출력
dilated_img = dilation_3x3(binary_img)
images = [img, dilated_img]
titles = ['Original image', 'Dilated_image 3x3 filter']
plot_img(images, titles)

# 침식(erosion) 결과 출력
erosion_img = erosion_3x3(binary_img)
images = [img, erosion_img]
titles = ['Original image', 'Erosion_image  3x3 filter']
plot_img(images, titles)


# 열기(open) 결과 출력
open_img = open_3x3(binary_img)
images = [img, open_img]
titles = ['Original image', 'Open_image  3x3 filter']
plot_img(images, titles)

# 닫기(close) 결과 출력
close_img = close_3x3(binary_img)
images = [img, close_img]
titles = ['Original image', 'Close_image  3x3 filter']
plot_img(images, titles)

# 팽창(dilation) 결과 출력
dilated_img = dilation_5x5(binary_img)
images = [img, dilated_img]
titles = ['Original image', 'Dilated_image 5x5 filter']
plot_img(images, titles)

# 침식(erosion) 결과 출력
erosion_img = erosion_5x5(binary_img)
images = [img, erosion_img]
titles = ['Original image', 'Erosion_image  5x5 filter']
plot_img(images, titles)


# 열기(open) 결과 출력
open_img = open_5x5(binary_img)
images = [img, open_img]
titles = ['Original image', 'Open_image  5x5 filter']
plot_img(images, titles)

# 닫기(close) 결과 출력
close_img = close_5x5(binary_img)
images = [img, close_img]
titles = ['Original image', 'Close_image  5x5 filter']
plot_img(images, titles)