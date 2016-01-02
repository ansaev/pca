#! /usr/bin/env python
from numpy import array, dot, mean, std, empty, argsort
from numpy.linalg import eigh, solve
from numpy.random import randn
from matplotlib.pyplot import subplots, show


def cov(data):
    N = data.shape[1]
    C = empty((N, N))
    for j in range(N):
        C[j, j] = mean(data[:, j] * data[:, j])
        for k in range(N):
            C[j, k] = C[k, j] = mean(data[:, j] * data[:, k])
    return C


def kaizer(pca_data, mean_nums, mean_vectors):
    arr_del = []
    arr_nums = mean_nums.tolist()
    arr_vectors = mean_vectors.tolist()
    arr_data = pca_data.tolist()
    for i in range(len(arr_nums)):
        if arr_nums[i] < 1:
            arr_del.append(i)
    arr_del.reverse()
    for del_num in arr_del:
        arr_nums.pop(del_num)
        for i in range(len(arr_vectors)):
            for j in range(len(arr_vectors[i])):
                if j == del_num:
                    arr_vectors[i].pop(j)
        for i in range(len(arr_data)):
            for j in range(len(arr_data[i])):
                if j == del_num:
                    arr_data[i].pop(j)
    return array(arr_data), array(arr_nums), array(arr_vectors)


def pca(data, pc_count=None):
    data -= mean(data, 0)
    data /= std(data, 0)
    C = cov(data)
    E, V = eigh(C)
    key = argsort(E)[::-1][:pc_count]
    E, V = E[key], V[:, key]
    U = dot(V.T, data.T).T
    return U, E, V


""" iris data """
import xlrd

book = xlrd.open_workbook("iris.xls")
sheet = book.sheet_by_index(0)
data_iris = []
for i in range(sheet.nrows):
    row = sheet.row_values(i)
    data_iris.append(row[0:-1])
data_iris = array(data_iris)
""" visualize """
trans_iris, E, V = pca(data_iris, 3)
# print('E', E)
# print('V', V)
# print("trans_iris", trans_iris)
trans_iris, E, V = kaizer(trans_iris, E, V)
print('E', E)
print('V', V)
print("trans_iris", trans_iris)
fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = subplots(1, 8)
ax1.scatter(data_iris[:50, 0], data_iris[:50, 1], c='r')
ax1.scatter(data_iris[50:100, 0], data_iris[50:100, 1], c='g')
ax1.scatter(data_iris[100:, 0], data_iris[100:, 1], c='b')

ax2.scatter(data_iris[:50, 1], data_iris[:50, 2], c='r')
ax2.scatter(data_iris[50:100, 1], data_iris[50:100, 2], c='g')
ax2.scatter(data_iris[100:, 1], data_iris[100:, 2], c='b')

ax3.scatter(data_iris[:50, 2], data_iris[:50, 3], c='r')
ax3.scatter(data_iris[50:100, 2], data_iris[50:100, 3], c='g')
ax3.scatter(data_iris[100:, 2], data_iris[100:, 3], c='b')

ax4.scatter(data_iris[:50, 0], data_iris[:50, 3], c='r')
ax4.scatter(data_iris[50:100, 0], data_iris[50:100, 3], c='g')
ax4.scatter(data_iris[100:, 0], data_iris[100:, 3], c='b')

ax5.scatter(data_iris[:50, 1], data_iris[:50, 3], c='r')
ax5.scatter(data_iris[50:100, 1], data_iris[50:100, 3], c='g')
ax5.scatter(data_iris[100:, 1], data_iris[100:, 3], c='b')

ax6.scatter(data_iris[:50, 0], data_iris[:50, 2], c='r')
ax6.scatter(data_iris[50:100, 0], data_iris[50:100, 2], c='g')
ax6.scatter(data_iris[100:, 0], data_iris[100:, 2], c='b')

ax7.scatter(trans_iris[:50, 0], trans_iris[:50, 0], c='r')
ax7.scatter(trans_iris[50:100, 0], trans_iris[50:100, 0], c='g')
ax7.scatter(trans_iris[100:, 0], trans_iris[100:, 0], c='b')

# ax8.scatter(trans_iris[:50, 0], trans_iris[:50, 2], c='r')
# ax8.scatter(trans_iris[50:100, 0], trans_iris[50:100, 2], c='g')
# ax8.scatter(trans_iris[100:, 0], trans_iris[100:, 2], c='b')
show()
