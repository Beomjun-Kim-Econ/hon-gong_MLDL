import numpy as np

            ### 4.1.1. ndarry 생성하기 ###
"""
배열을 생성하는 쉬운 방법은 array 함수를 이용하는 것이다.
The most easy way to generate an array is to use method 'array'.

파이썬 기본 데이터 구조 역시 넘파이 배열로 변환할 수 있다.
it's possible from Python vanila data structure like list to numpy array. 
"""
# i.e.
data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)
print(arr1)
# [[6, 7.5, 8, 0, 1]]

"""
리스트 길이가 동일한 중첩된 순차 데이터는 다차원 배열로 변환 가능하다.
다시 말해 2차원 이상의 리스트에서, 길이 조건이 맞다면 넘파이 2차원 배열(행렬, matrix) 
또는 그 이상이면 텐서 tensor로 변환 가능하다.
"""
# i.e.
data2 = [[1,2,3,4], [5,6,7,8]]
arr2 = np.array(data2)
print(arr2)

"""
각 배열은 차원(2차원 - 행렬 matrix, 3차원 이상 - 텐서 tensor 의 그 차원) dimension 을 갖는다. 이는 ndim으로 확인.
그리고 따로 설정하지 않으면 넘파이가 자료형도 알아서 적절하게 추론한다.
Numpy will infer the most apt datatype if user didn't set that.
"""
print(arr2.ndim)
# 2 2차원(행렬)이니까.

    ### 배열 생성 함수 method (function) for generating array ###

# array <- data1, data2의 예시와 같은 경우 깊은 복사 (deep copy)한다.
# asarray <- 이미 넘파이 배열이면 복사를 하지 않는다. 즉 원본 데이터를 참조한다. 넘파이 배열이 아니면 array랑 똑같이 복사한다.
# arange <- 내장 range와 동일. 단 1xn 꼴의 넘파이 배열로 리턴한다.

# ones <- 주어진 dtype과 모양을 가지는 배열 생성. 내용은 1로 초기화.
# ones_like <- 주어진 dtype, shape와 동일한 배열 생성. 내용은 1로 초기화.
# i.e.
array1 = np.ones((2, 3))   # Creates a 2x3 array of ones
print(array1)
# Output:
# [[1. 1. 1.]
#  [1. 1. 1.]]
array2 = np.array([[3, 5, 7], [2, 4, 6]])
array3 = np.ones_like(array2)  # Creates an array with the same shape as array2 filled with ones
print(array3)
# Output:
# [[1 1 1]
#  [1 1 1]]

# zeros, zeros_like <- 위 둘과 동일, 단 내용이 0으로 초기화.
# empty, empty_like <- 메모리를 새로 할당받아 새 배열 생성. 단, 값 초기화는 하지 않는다. 내부 값은 랜덤.
# i.e.
array4 = np.array([[1, 2, 3], [4, 5, 6]])
array5 = np.empty_like(array2)  # Creates an array with the same shape as array2 but uninitialized
print(array5)
# Output (values will be arbitrary and vary each time):
# [[16777472        0 602937285]
#  [       0 602937283 602937289]]
# full, full_like <- full은 인수로 shape와 숫자 하나를 받는다. 그 shpae 행렬에 받은 숫자로 채운다. full_like는 자명하다.
# eye.identity <- 크기가 n인 단위행렬 (identity matrix) 생성