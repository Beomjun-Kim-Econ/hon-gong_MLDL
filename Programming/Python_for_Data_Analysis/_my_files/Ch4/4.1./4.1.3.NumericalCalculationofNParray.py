import numpy as np

            ### 4.1.3. 넘파이 배열의 산술 연산 Numerical Calculation of numpy array ###

"""
몇 가지 산술 연산을 살펴보자.
"""
arr = np.array([[1,2,3],[4,5,6]], dtype = np.float64)

# array multiplication
arr_mul = arr*arr
print('arr multiplicaion: \n', arr_mul)

# array addition / substraction / scalar multiplication / reciprocal
arr_add = arr + arr
arr_sub = arr - arr
arr_scl_mul = 3 * arr
arr_rep = 1 / arr
print('arr addition: \n', arr_add)
print('arr substraction: \n', arr_sub)
print('arr scalar multiplication: \n', arr_scl_mul)
print('arr reciprocal: \n', arr_rep)

# comparing - boolean 비교 - 불리언
arr2 = np.array([[0,4,1], [7,2,12]], dtype=np.float64)
arr_bool = arr2 > arr
print(arr_bool)