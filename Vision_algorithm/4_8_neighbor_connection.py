# -*- coding: utf-8 -*-
# 4연결성과 8연결성 라벨링


def four_connect(img):
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray_img = np.pad(gray_img, (1,1), mode = 'constant', constant_values = 0)
  gray_img = np.where(gray_img != 0 , -1, 0)

  h, w = gray_img.shape
  answer = np.zeros(shape=(h, w))
  label = 1

  for i in range(1, h-1) : 
    for j in range(1, w - 1) :
      if gray_img[i][j] == -1 :
        find_4(gray_img,answer, i,j,label, h, w)
        label += 1

  answer = answer[1:-1, 1: -1] # padding 제거  
  print(f'Total Label = {label-1}') # label-1개가 전체 label 값  
  random_color = np.random.rand(3,label-1) # random color value, label마다 색을 다르게 하기 위해 R,G,B 마다 다른 값을 생성하도록 한다.
  k_1 = np.array([[random_color[0][int(x)-1]* x if int(x) != 0  else x for x in row ] for row in answer])
  k_2 = np.array([[random_color[1][int(x)-1]* x if int(x) != 0  else x for x in row ] for row in answer])
  k_3 = np.array([[random_color[2][int(x)-1]* x if int(x) != 0  else x for x in row ] for row in answer])

  scaling_1 = (k_1- k_1.min()) / (k_1.max() - k_1.min()) # float이기 때문에 [0,1] 범위를 맞춰줌
  scaling_2 = (k_2- k_2.min()) / (k_2.max() - k_2.min())
  scaling_3 = (k_3- k_3.min()) / (k_3.max() - k_3.min())

  result = np.dstack((scaling_1,scaling_2,scaling_3)) # GRAY2RGB

  return result, label-1

def find_4(array,answer,i, j, label,h, w) :
  if (array[i][j] == -1) and (( i> 0 and i < h-1 ) and (j > 0  and i < w -1 )) :
    answer[i][j] = label
    array[i][j] = 0
   
    find_4(array,answer, i, j+1, label, h, w) #오
    find_4(array,answer, i-1, j , label, h, w) #위
    find_4(array,answer, i, j-1, label, h, w) # 왼 
    find_4(array,answer,i+1, j, label, h, w) # 아래


def eight_connect(img):
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray_img = np.pad(gray_img, (1,1), mode = 'constant', constant_values = 0)
  gray_img = np.where(gray_img != 0 , -1, 0)

  h, w = gray_img.shape
  answer = np.zeros(shape=(h, w))
  label = 1

  for i in range(1, h-1) : 
    for j in range(1, w - 1) :
      if gray_img[i][j] == -1 :
        find_8(gray_img,answer, i,j,label, h, w)
        label += 1

  answer = answer[1:-1, 1: -1] # padding 제거  
  print(f'Total Label = {label-1}') # label-1개가 전체 label 값  
  random_color = np.random.rand(3,label-1) # random color value, label마다 색을 다르게 하기 위해 R,G,B 마다 다른 값을 생성하도록 한다.
  k_1 = np.array([[random_color[0][int(x)-1]* x if int(x) != 0  else x for x in row ] for row in answer])
  k_2 = np.array([[random_color[1][int(x)-1]* x if int(x) != 0  else x for x in row ] for row in answer])
  k_3 = np.array([[random_color[2][int(x)-1]* x if int(x) != 0  else x for x in row ] for row in answer])

  scaling_1 = (k_1- k_1.min()) / (k_1.max() - k_1.min()) # float이기 때문에 [0,1] 범위를 맞춰줌
  scaling_2 = (k_2- k_2.min()) / (k_2.max() - k_2.min())
  scaling_3 = (k_3- k_3.min()) / (k_3.max() - k_3.min())

  result = np.dstack((scaling_1,scaling_2,scaling_3)) # GRAY2RGB

  return result, label-1

def find_8(array,answer,i, j, label,h, w) :
  if (array[i][j] == -1) and (( i> 0 and i < h-1 ) and (j > 0  and i < w -1 )) :
    answer[i][j] = label
    array[i][j] = 0
   
    find_8(array,answer, i, j+1, label, h, w) #오
    find_8(array,answer, i-1, j+1, label, h, w) # 오 위
    find_8(array,answer, i+1, j+1, label, h, w) #오 아래 
    find_8(array,answer, i-1, j , label, h, w) #위
    find_8(array,answer, i, j-1, label, h, w) # 왼
    find_8(array,answer, i-1, j-1, label, h, w) #왼 위
    find_8(array, answer,i+1, j-1, label, h, w) #왼 아래  
    find_8(array,answer,i+1, j, label, h, w) # 아래




if __name__ == '__main__':
  try : 
    img = cv2.imread('sample.png')
    # image 출력
    print("실제 이미지 출력")
    plt.title("실제 이미지 출력")
    plt.imshow(img)
    plt.show()

    img = cv2.imread('sample.png')
    labeled_img, count_label = four_connect(img)

    # image 출력
    plt.title(f"Four Connect Result, Total Label : {count_label}")
    plt.imshow(labeled_img)
    plt.grid(None)   
    plt.xticks([])
    plt.yticks([])
    plt.show()

    img = cv2.imread('sample.png')

    labeled_img, count_label = eight_connect(img)

    # image 출력
    plt.title(f"Eight Connect Result, Total Label : {count_label}")
    plt.imshow(labeled_img)
    plt.grid(None)   
    plt.xticks([])
    plt.yticks([])
    plt.show()

  except : 
    print("파일이 없습니다 ")
    print('!wget https://i.imgur.com/iWGMlJR.png')
    print("!mv iWGMlJR.png sample.png ")
    print("를 실행해 주세요")

