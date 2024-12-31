import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fruits = np.load('/Users/beomjunkim/Programming/Hongong_MLDL/data/fruits_300.npy')
print(fruits.shape)
# (300, 100, 100) -> (n번째 샘플, 각 이미지의 높이, 각 이미지의 너비)
print(fruits[0, 0, :])

# Visualization
plt.imshow(fruits[0], cmap='gray')
plt.show()

# plt.imshow(fruits[0], cmap='gray_r')
# plt.show()

fig, axs = plt.subplots(1,2)
axs[0].imshow(fruits[100], cmap = 'gray_r')
axs[1].imshow(fruits[200], cmap = 'gray_r')
plt.show()


# Analysis for pixel values

# for each image...
apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)
print(apple.shape) # There're 100 rows. And each row corresponds to each image.
print(apple.mean(axis = 1)) # The means of each image

graphs, each_graph = plt.subplots(2,2)
each_graph[0][0].hist(np.mean(apple, axis = 1), alpha = 0.8, color = 'red')
each_graph[0][1].hist(np.mean(pineapple, axis = 1), alpha = 0.8, color = 'green')
each_graph[1][0].hist(np.mean(banana, axis = 1), alpha = 0.8, color = 'yellow')

each_graph[1][1].hist(np.mean(apple, axis = 1), alpha = 0.5, color = 'red')
each_graph[1][1].hist(np.mean(pineapple, axis = 1), alpha = 0.5, color = 'green')
each_graph[1][1].hist(np.mean(banana, axis = 1), alpha = 0.5, color = 'yellow')
plt.show()

# for each pixel...
# fig, axs = plt.subplots(1, 3, figsize = (20, 5))
# axs[0].bar(range(10000), np.mean(apple, axis = 0))
# axs[1].bar(range(10000), np.mean(pineapple, axis = 0))
# axs[2].bar(range(10000), np.mean(banana, axis = 0))
# plt.show()

apple_mean = np.mean(apple, axis = 0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis = 0).reshape(100, 100)
banana_mean = np.mean(banana, axis = 0).reshape(100, 100)
fig, axs = plt.subplots(1, 3)
axs[0].imshow(apple_mean, cmap = 'gray')
axs[1].imshow(pineapple_mean, cmap = 'gray')
axs[2].imshow(banana_mean, cmap = 'gray')
plt.show()

# choosing the nearest picture from mean
# let me pick the nearest 100 pictures from apple_mean
abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis = (1,2))
print(abs_mean.shape)

apple_index = np.argsort(abs_mean)[:100]        # argsort: ascending sort
fig, axs= plt.subplots(10, 10, figsize = (10, 10))
for i in range(10):
    for j in range(10):
        axs[i, j].imshow(fruits[apple_index[i*10 + j]], cmap = 'gray')
        axs[i, j].axis('off')   # 축 숨기기

plt.show()