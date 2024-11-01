import numpy as np
                ### 4.1.7. Transpose and Changing Axis ###

"""
행렬(배열)의 전치(transpose): .T 를 사용한다.
"""
arr = np.arange(15).reshape((3,5))
print(arr)  # 3*5 행렬
print(arr.T)

"""
행렬(배열)의 곱 내지는 내적(dot product). np.dot 이나 @을 사용한다.
"""
arr2 = np.array([[0,1,0], [1,2,-2], [6,3,2], [-1,0,1], [1,0,1]])
print(arr2)     # 5*3 행렬
result = np.dot(arr, arr2)
result2 = arr @ arr2
print(result)   # arr과 arr2 곱
print(result2)  # arr과 arr2 곱

"""
transpose 는 2차원(행렬)에서나 가능하다. 일반적으로축을 바꿔보자.: swapaxes 를 이용하자.
"""
print(arr)
print(arr.swapaxes(0,1))    # 인수를 잘 보세요. 0번째/1번째 축을 뒤바꿈. ; 여기선 transpose와 동일
