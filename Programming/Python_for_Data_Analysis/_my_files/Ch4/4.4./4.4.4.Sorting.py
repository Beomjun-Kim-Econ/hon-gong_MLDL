import numpy as np

            ### 4.4.4. Sorting ###

rng = np.random.default_rng()
"""
넘파이에서도 sort 메서드를 사용해서 정렬할 수 있다.
주의 !!! .sort() 는 None을 리턴한다. 뭔가 할당하지 말고, arr.sort() 이렇게 해서 arr 자체를 sort 하고, arr을 그냥 쓰자.
만약 원본 arr을 살리고 싶다면 copy를 해야한다.
"""
arr = rng.standard_normal(6)
arr.sort()
print(arr) # 정상
result = arr.sort()
print(result)   # None 리턴

"""
2차원 이상 배열에서 역시 축을 설정하여 부분적으로 정렬할 수 있다.
"""
arr2 = rng.standard_normal((5,3))
print(arr2)
arr2.sort(axis=0)   #위에서 아래로 점점 커진다. ; 행을 지정 -> 열을 기준으로 변화
# 만약 axis=1 이라면 왼쪽에서 오른쪽으로 점점 커진다.
print(arr2)
