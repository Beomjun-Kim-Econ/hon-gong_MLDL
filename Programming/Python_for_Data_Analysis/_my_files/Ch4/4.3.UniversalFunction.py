import numpy as np
import random as rd
                ### 4.3. Universial Functions ###

"""
유니버셜 함수 Universal function 이란 ndarray 안의 데이터 원소 별로 연산을 수행하는 함수다.
줄여서 ufnc 라고도 한다. 하나 이상의 스칼라 값을 받아 하나 이상의 스칼라 결과값을 반환하는, 
간단한 함수를 빠르게 수행하는 벡터화된 래퍼(wrapper) 함수 라고 할 수 있다. 
*래퍼함수: 실제 함수(original function)를 감싼 함수(wrapper function)로, 
실제 함수 호출 시 특별한 동작을 하도록 기능을 덧붙인 함수
"""
arr = np.arange(10)
arr_sqrt = np.sqrt(arr)     # ufnc1: sqrt
print(arr_sqrt)
arr_exp = np.exp(arr)       # ufnc2: exp
print(arr_exp)

"""
앞의 함수들은 unary (단일체의, 단항의) universal function 이다. 
numpy.add, numpy.maximym 처럼 2개의 매개변수를 받아 단일 배열을 함수는 binary ufnc 이다.
"""
rng = np.random.default_rng()
x = rng.standard_normal(8)
y = rng.standard_normal(8)
print(np.maximum(x,y))

"""
흔치 않지만, 여러개의 배열을 변환하는 경우 역시 있긴 하다. np.modf 가 대표적으로, math.modf 의 벡터화 버전이다.
이는 분수를 받아 몫과 나머지를 함께 반환한다.
"""
arr2 = rng.standard_normal(7) * 5
print(arr2)
remainder, whole_part = np.modf(arr2)
print(remainder)
print(whole_part)

"""
선택적으로 out 인수를 사용하여 계산 결과를 새로운 배열로 만들지 않고 기존 배열에 할당할 수 있다.
"""
a = np.array([1, 2, 3])
# 새로운 배열을 만드는 경우
tmp = np.add(a, 1)
print(tmp)

# out인수를 사용해보자
out_array = np.zeros_like(a)
np.add(a, 1, out=out_array)
print(out_array)

