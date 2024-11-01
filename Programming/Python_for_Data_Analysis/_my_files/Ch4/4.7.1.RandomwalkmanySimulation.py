import numpy as np
import matplotlib.pyplot as plt


                ### 4.7.1. Random Walk many simulation ###
rng = np.random.default_rng()
"""
랜덤워크 시뮬레이션을 5000번 정도 한다고 해보자. 4.7. 의 코드를 조금 수정해서 할 수 있다.
np.random 함수에 크기가 2인 튜플을 넘기면 2차원 배열이 생성되고, 각 열에서 누적합을 구해서 5000회의 시뮬레이션을 한번에 처리할 수 있다.
"""
nwalks = 5_000
nstpes = 1_000
draws = rng.integers(0, 2, size=(nwalks, nstpes))
# 0 또는 1, 행렬의 크기는 5000 * 1000. 즉, 각 행이 한번의 시행인 것이다. 그렇게 5000개의 행이 있지 않겠는가?
steps = np.where(draws > 0, 1, -1)
walks = np.cumsum(steps, axis=1)
print(walks)    # 이것의 마지막 값이 각각 최종 1000번째, 즉 랜덤워크 시행의 결과임.

# 최대 최소를 구해보자
print('max:', walks.max())
print('min:', walks.min())

"""
누적합이 -30 혹은 30이 되는 최소 지점을 계산해보자.
"""
hits30 = (np.abs(walks) >= 30).any(axis=1)      #우선 누적합이 30 혹은 -30을 찍기는 하는 지 체크
print(hits30)

hits30_count = hits30.sum()     # 5000번의 시행 중, 한 번의 시행에서 누적합이 30 혹은 -30을 한번이라도 찍어본 경우가 몇번인가?
print(hits30_count)

# 나머지 생략...