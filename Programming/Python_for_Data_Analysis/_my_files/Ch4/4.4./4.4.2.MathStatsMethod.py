import numpy as np

            ### 4.4.2. Mathematical / Statistical Method ###

"""
배열 전체, 혹은 배열의 한 축에 속하는 자료에서 통계를 계산하는 수학 함수는 배열 클래스 매서드로 사용할 수 있다.
sum, mean, std 등은 넘파이의 최상위 함수를 이용하거나, 배열의 인스턴스 메서드를 사용해서 구한다.
*인스턴스 메서드: 가장 일반적인 그 메서드
"""
rng = np.random.default_rng()
arr = rng.standard_normal((5,4))
print(arr)
print(arr.mean())
print(np.mean(arr))
print(arr.sum())

"""
배열의 축을 설정하여 축(예컨데 2차원 배열, 즉 행렬이라면 행과 열)을 설정하여 구할 수도 있다.
기본적으로 0은 행, 1은 열이다.
"""
print(arr.mean(axis = 1)) # 각 열마다의 평균 반환
print(arr.sum(axis = 0)) # 각 행마다의 sum 반환

"""
cumsum 함수와 cumprod 함수는 각각 cumulative summation, cumulative product(곱) 을 의미한다.
"""
arr = np.arange(1,8,1)
print(arr.cumsum())
print(arr.cumprod())

"""
cumsum, cumprod를 2차워 배열에 응용해보자. 역시 배열의 축을 설정하여 행별, 열별 누적합/누적곱 '행렬'을 구할 수도 있다.
다시말해 cumsum의 결과 역시 원래 행렬과 shape가 같은 행렬이라는 것이다.
"""
arr2 = np.array([[0,1,2], [3,4,5], [6,7,8]])
print(arr2)
print(arr2.cumsum(axis=0)) # 행의 누적합
print(arr2.cumsum(axis=1)) # 열의 누적합


