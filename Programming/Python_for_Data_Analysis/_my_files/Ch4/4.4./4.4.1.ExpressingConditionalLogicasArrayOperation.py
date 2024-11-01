import numpy as np

            ### 4.4.1.Expressing Conditional Logic as Array Operation ###

"""
배열을 이용하여 조건부 conditional 을 표현해보자.
"""
xarr = np.arange(1.1, 1.6, 0.1)
yarr = np.arange(2.1, 2.6, 0.1)
cond = np.array([True, False, True, True, False])
result = [(x if c else y)
          for x, y, c in zip(xarr, yarr, cond)]
print(result)
"""
위 방법은 좋지만 두가지 문제가 있다. 첫재, 빌트인 함수를 써서 느리다. 둘째, 다차원에서는 이용이 불가능하다.
고로 넘파이를 도입하자. np.where를 사용하자.
"""
rng = np.random.default_rng(seed = 12345)
arr = rng.standard_normal((4,4))
print(arr)
print(arr > 0)
tmp = np.where(arr > 0, 2, -2)
print(tmp)

"""
위의 방법에서 스칼라 말고 array를 사용할 수도 있다.
"""
arr2 = rng.standard_normal((4,4))
print(arr2)
tmp2 = np.where(arr2 > 0, 2, arr2) # 0 보다 크면 2, 아니면 기존 유지
print(tmp2)
# 위 자리에 arr2 가 아니라 arr2와 차원이 같은 배열 matx를 넣으면 "기존유지"가 아니라 matx로 대체된다.
