import numpy as np
                ### 4.1.6. Fancy Indexing ###

"""
팬시 색인은 정수 배열을 사용한 색인을 설명하기 위해 차용한 단어이다.
"""

arr = np.zeros((8,4))
for i in range(8):
    arr[i] = i
print(arr)

"""
특정 행의 하위 집합을 선택하고 싶다면 순서가 명시된 ndarry나 리스트를 주자
"""
print(arr[[2,3,5]])

"""
다차원 색인 배열을 보자
"""
arr2 = np.arange(32).reshape((8,4))     # 당연히 reshape 에 들어가는 숫자 개수 만큼 arange 인수로 들어가야함
print(arr2)
print(arr2[[1,5,7,2],[0,3,1,2]])    #(1,0), (5,3), (7,1), (2,2)에 해당하는 결과 리턴
print(arr2[[1,5,7,2]][:,[0,3,1,2]]) #1,5,7,2 번째 row를 :, 즉 열로 세우고, 그 순서는 0,3,1,2 순서


