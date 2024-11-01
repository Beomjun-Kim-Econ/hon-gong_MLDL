import numpy as np
                ### 4.2. Generating Random number ###

"""
난수를 발생시켜보자. 우선 standard normal distribution 을 보지.
"""
samples = np.random.standard_normal(size = (4,4))
print(samples)

"""
파이썬 built-in module인 random을 이용해보자.
"""
import time
from random import normalvariate
N = 1_000_000_0
start_time = time.time()
samples = [normalvariate(0, 1) for _ in range(N)]
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")

"""
넘파이가 압도적으로 더 빠르다.
"""
start_time_np = time.time()
samples = np.random.standard_normal(N)
end_time_np = time.time()
execution_time_np = end_time_np - start_time_np
print(f"NP Execution time: {execution_time_np:.6f} seconds")

"""
위의 난수는 엄밀하게는 난수는 아니고 유사난수 pseudorandom 이라고 한다.
난수 생성기의 시드 seed 값에 따라 정해진 난수를 알고리듬으로 생성하기 때문이다.
즉 시드를 명시적으로 설정할 수 있다.
"""
rng = np.random.default_rng(seed=12345)
data = rng.standard_normal(size=(2,3))
print(data)
