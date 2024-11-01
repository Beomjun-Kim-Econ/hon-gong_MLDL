import numpy as np

            ### 4.1.4. 색인과 슬라이싱 기초 Indexing and Slice slicing ###

"""
1차원 배열은 파이썬 기본 리스트와 유사하게 작동한다.
"""
# i.e.
arr1 = np.arange(10)
print(arr1[5])  #5
print(arr1[5:8])    # array([5 6 7])
arr2 = arr1.copy()
arr2[5:8] = 12
print(arr2)     # 5~7번째가 12가 되어있음


"""
슬라이스 함수 slice 는 카피하지 않고 원본 데이터를 참조한다.
카피하고 싶으면 .copy() 를 꼭 사용하자.
"""
# i.e.
arr3 = np.arange(10)
sliced_arr3 = arr3[5:8]
sliced_arr3[0:2] = 12
print('sliced_arr3: ', sliced_arr3)
print('arr3: ', arr3) # 원본 데이터도 바뀜. 카피가 아니라 참조 했기 때문.

"""
다차원 배열에서의 인덱스는 2중, 3중 리스트와 비슷하다.
"""
arr4 = np.array([[1,2,3], [4,5,6]])
print(arr4[0]) #[1 2 3]
print(arr4[0][1]) # 2

arr3d = np.array([[[1,2,3], [4,5,6]],[[7,8,9], [10,11,12]]])
print(arr3d, '\n\n\n', arr3d[0])

"""
다차원 배열에서, 내부 배열에는 스칼라값, 배열 모두 할당할 수 있다. 
"""
old_values = arr3d[0].copy()
arr3d[0] = 42
print(arr3d)
arr3d[0] = old_values
print(arr3d)
print(arr3d[1][0])
print(arr3d[1,0]) # arr3d[1][0]과 arr3d[1,0]은 같다. 단 후자는 넘파이 전용

"""
다차원에서 슬라이스로 선택할 수도 있다. 행렬을 떠올리자.
"""
arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(arr2d[:2])
print(arr2d[:2, 1:])
lower_dim_slice = arr2d[1, :2]
print(lower_dim_slice)
#lower_dim_slice는 1차원이고 축 크기가 하나인 튜플

