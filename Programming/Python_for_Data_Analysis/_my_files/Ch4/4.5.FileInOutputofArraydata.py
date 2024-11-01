import numpy as np
                ### 4.5.File Input and Output of Array data ###

"""
넘파이는 디스크에서 텍스트나 바이너리 형식의 데이털르 불러오거나 저장할 수 있다. 
사실 대부분의 경우 텍스트나 표 형식의 데이터를 쓰는데, 이건 보통 판다스나 다른 툴을 이용한다.
여기서는 가볍게 넘파이의 이진 형식만을 살펴보자.

np.save, np.load 는 배열 데이터를 효과적으로 디스크에 저장하고 불러오는 함수이다.
배열은 기본적으로 압축되지 않은 raw binary data 인 .npy 형식으로 저장된다. 
.npy는 np.load를 통해 불러올 수 있다.
"""
arr = np.arange(10)
# np.save("some_array", arr)    # 자동으로 .npy 확장자 추가
a = np.load("some_array.npy")
print(a)

"""
np.savez 를 이용하여 .npz 파일을 만들 수도 있다. .npz 는 여러개의 배열을 '딕셔너리'형태로 저장한다.
"""
arr2 = np.arange(7)
np.savez("array_archive.npz", a = arr, b = arr2)
arch = np.load("array_archive.npz")
print(arch["b"])

"""
위의 savez 에서 압축이 잘 되는 형식의 데이터라면 numpy.savez_compressed 를 사용할 수도 있다.
"""
