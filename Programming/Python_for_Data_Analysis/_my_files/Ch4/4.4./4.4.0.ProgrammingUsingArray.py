import numpy as np

            ### Programming using array ###

"""
numpy array 는 loop를 줄일수 있다. 또한 코드의 길이 역시 concise 할 수 있다. 속도도 수십배 빠르다.
loop 계산을 numpy array 로 바꿔 계산하는 것을 vectorization이라 한다.
간단하게 sqrt(x^2 + y^2)를 계산하는 예시를 보자. 
여기서 meshgrid는 2개의 1차원 배열을 받아 모든 x, y 짝을 만들 수 있는 2차원 배열 2개를 반환한다. 
"""
points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points, points)
# meshgrid의 xs는 row가 변화 변화 변화, ys는 column이 변화 변화 변화

"""
그리드 위에서 두점을 갖고 단단히 계산해보자.
"""
z = np.sqrt(xs**2 + ys**2)
print(z)
"""
z를 시각화해보자
"""
import matplotlib.pyplot as plt
plt.imshow(z, cmap=plt.cm.gray, extent=[-5,5,-5,5])
plt.colorbar()
plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
plt.show()

