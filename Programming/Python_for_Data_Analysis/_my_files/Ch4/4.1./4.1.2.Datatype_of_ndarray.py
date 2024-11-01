import numpy as np

            ### 4.1.2. ndarray의 자료형 datatype ###

"""
자료형 dtype은 ndarray가 메모리에 있는 특정 데이터를 해석하는 데에 필요한 정보 또는 메타데이터를 담고 있는 특수한 객체(object)다.
"""
# i.e.
arr1 = np.array([1,2,3], dtype = np.float64)
arr2 = np.array([1,2,3], dtype = np.int32)
print(arr1.dtype)   # float64
print(arr2.dtype)   # int32

"""
astype 을 이용해 dtype을 다른 형으로 명시적으로 변환(cast)가 가능하다.
카피도 한다. 그러므로 원본 배열의 데이터타입을 건드리지는 않는다.(애초에 딥카피 하니까)
str도 float, int 등으로 변환 가능하다.
변환이 부적절하면 ValueError 가 발생한다.
"""
# i.e
print('origianl dtype: ', arr1.dtype)
new_arr = arr1.astype(np.int64)
print('new dtype: ', new_arr.dtype)
str_arr = np.array(['1', '2', '3'])
new_str_arr = str_arr.astype(np.int64)
print('new str arr dtype: ', new_str_arr.dtype)

