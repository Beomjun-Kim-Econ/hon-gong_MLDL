import numpy as np
import matplotlib.pyplot as plt


                ### 4.7. Random Walk ###

"""
랜덤워크 예제를 만들어보자. 계단의 중간(0)에서 한 계단 올라가거나(+1) 내려간다(-1).
우선 순수 파이썬 내장 모듈인 ramdom 을 이용해보자.
"""
import random
position = 0
walk = [position]
nsteps = 1000
for i in range(nsteps) :
    step = 1 if random.randint(0,1) else -1     # 불리안에서 1이 True, 0이 False 임을 활용한 것
    position += step
    walk.append(position)

plt.plot(walk[:100])
plt.show()

"""
위를 기준으로 간단하게 오르내린 위치의 최소값/최댓값 같은 간단한 통계를 구할 수도 있다.
"""
print(min(walk))
print(max(walk))
"""
심지어 특정 위치에 도달하기까지의 시간을 구할 수도 있다. 
우선 계단의 처음 위치(0)에서 처음으로 10칸 떨어지기까지 얼마나 걸렸는지 확인해보자.
"""
time = (np.abs(walk) >= 10).argmax()
print(time)