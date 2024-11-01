import numpy as np

            ### 4.4.3. Method for Boolean ###

rng = np.random.default_rng()
"""
불리언의 값은 1(True), 0(False)로 강제된다. 즉 불리언에서 Sum 메서드를 실행하면 True의 개수를 반환하는 거나 다름 없다.
"""
arr = rng.standard_normal(100)
num_of_true = (arr > 0).sum()
num_of_false = (arr < 0).sum()
print(num_of_true)
print(num_of_false)
"""
any, all 메서드는 불리안에서 유용하다.
any는 하나 이상의 True가 있는지 검사하고, all은 모든 원소가 True 인지 검사한다.
이 두 매서드는 이렇게 불리안 타입이 아닌 경우에도 쓸수 있는데, 이 경우 0이면 False, 0이 아니면 True 로 들어온다.
"""
bools = np.array([True, True, False, True, False])
print(bools.any())
print(bools.all())
bools_special = np.array([1,2,1])
print(bools_special.all())
