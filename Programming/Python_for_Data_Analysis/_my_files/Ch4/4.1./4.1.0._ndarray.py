import numpy as np

            ### 4.1. Array 배열 ###

"""
is almost similar with Matrix
거의 행렬과 동일하다.
"""
# i.e.
arr = np.array([[1,2],[3,4]])
print(arr)
# [[1 2]    first row 첫번째 행
# [3 4]]    second row 두번째 행
# 이런 방식의 2차원(dimension) 행렬 뿐 아니라, 3차원 이상의 텐서 tensor 역시 가능하다.
"""
기본적인 스칼라배, 행렬합 모두 가능하다
"""
# i.e.
print(arr + arr)
# [[2 4]
# [6 8]]

print(3*arr)
# [[3 6]
# [9 12]]

"""
모든 배열은 
(1) 그 shape(2x2인지, 3x4인지, 3차원 텐서 tensor라면 2x3x3인지 알려줌. tuple)
(2) dtype(배열에 저장된 자료형을 알려줌)
을 갖는다.  
"""
# i.e.
print(arr.shape)
# (2, 2)
print(arr.dtype)
# int64