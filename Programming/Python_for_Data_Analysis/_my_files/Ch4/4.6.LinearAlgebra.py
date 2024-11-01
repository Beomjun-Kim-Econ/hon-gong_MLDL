import numpy as np

                ### 4.6.Linear Algebra ###
rng = np.random.default_rng()
"""
product of matrics, slicing, determinant, operation of square matrix 등은 중요한 토픽이다.
우선 행렬곱은 np.dot를 이용한다. 
"""
x = np.array([[1,2,3], [4,5,6]], dtype=float)   #2*3 행렬
y = np.array([[6,23], [-1,7], [8,9]], dtype=float)  # 3*2 행렬
result = x.dot(y)   # xy 이다. yx가 아니다.
result_same = np.dot(x, y) # 마찬가지로 xy 이다. 윗 식과 동일하다.
resutl_same2 = x @ y # 역시 마찬가지로 xy 이다.
print(result, "\n", result_same, "\n", resutl_same2)

"""
넘파이의 linalg 는 역행렬, 행렬식 등 여러 행렬 관련 함수가 있다.
"""
x = rng.standard_normal((5, 5))
mat = x.T @ x
inverse = np.linalg.inv(mat)
print(mat)
print(inverse)




