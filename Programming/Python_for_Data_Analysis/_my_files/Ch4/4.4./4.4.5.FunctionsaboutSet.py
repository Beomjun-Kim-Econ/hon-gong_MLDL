import numpy as np

            ### 4.4.5.Functions about Set ###

rng = np.random.default_rng()
"""
넘파이에서도 집합 연산을 제공한다. 가장 자주 쓰이는 것은 중복 원소를 제거하고 남은 것들을 정렬까지 해주는 np.unique 이다.
np.unique 는 sort와 달리 None을 리턴하는 건 아니고 우리 생각 그대로 움직인다.
"""
names = np.array(["Bob", "Will", "Joe", "Bob", "Will", "Joe", "Joe"])
print(np.unique(names))
ints = np.array([3,3,3,2,2,1,1,4,4])
print(np.unique(ints))
#순수 파이썬:
print(sorted(set(names)))
# 당연히 넘파이가 훨씬 빠르고, 넘파이 버전은 ndarray 를 반환한다.

"""
np.in1d 는 인수로 받은 배열의 원소가 기존 배열에 포함되는지를 검사하고 이를 불리언으로 리턴한다.
"""
values = np.array([6,0,0,3,2,5,6]) # 기존 배열
result = np.in1d(values, [2,3,6])   # values의 각 자리가 2,3,6이 맞는치 체크
print(result)


