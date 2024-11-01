import numpy as np
                ### 4.1.5. Selection by Boolean ###

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.array([[4,7],[0,2],[-5,6],[0,0],[1,2],[-12,-4],[3,4]])

"""
위의 데이터를 바탕으로 인덱싱 할 수 있다.
"""
print(names=='Bob')
print(data[names=='Bob'])


"""
일반 인덱스도 혼용 가능하다.
"""
print(data[names == 'Bob', 1:])
print(data[names == 'Bob', 1])


"""
!=이나 ~을 사용할 수도 있다. ~는 특히 단순 뒤집기에 유리하다.
"""
print(names != 'Bob')
print(~(names == 'Bob'))    # 당연히 바로 윗줄과 동일한 의미
print(data[~(names == 'Bob')])
cond = (names == 'Bob')
print(data[~cond])  # 뒤집기에 특화


"""
AND와 OR은 &과 |로 처리한다. 불리언에서는 and, or이 불가능하다!!!!
"""
mask = (names == 'Bob') | (names == 'Will')
print(mask) # Bob 과 Will 모두에 True


"""
불리언 배열에 값을 대입하면 원본 값을 조정할 수 있다. 예제로 보자.
"""
data_copy1 = np.copy(data)
data_copy1[data_copy1 < 0] = 0
print(data_copy1)
data_copy2 = np.copy(data)
data_copy2[names != 'Joe'] = 7
print(data_copy2)
