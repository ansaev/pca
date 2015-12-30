#! /usr/bin/env python
from numpy import array, dot, mean, std, empty, argsort
from numpy.linalg import eigh, solve
from numpy.random import randn
from matplotlib.pyplot import subplots, show


def cov(data):
    """
        covariance matrix
        note: specifically for mean-centered data
    """
    N = data.shape[1]
    C = empty((N, N))
    for j in range(N):
        C[j, j] = mean(data[:, j] * data[:, j])
        for k in range(N):
            C[j, k] = C[k, j] = mean(data[:, j] * data[:, k])
    return C


def pca(data, pc_count = None):
    """
        Principal component analysis using eigenvalues
        note: this mean-centers and auto-scales the data (in-place)
    """
    data -= mean(data, 0)
    data /= std(data, 0)
    C = cov(data)
    E, V = eigh(C)
    key = argsort(E)[::-1][:pc_count]
    E, V = E[key], V[:, key]
    U = dot(V.T, data.T).T
    return U, E, V

""" test data """
data = array([randn(5) for k in range(100)])
data[:50, 2:4] += 5
data[50:, 2:5] += 5

""" iris data """
import xlrd
book = xlrd.open_workbook("iris.xls")
sheet = book.sheet_by_index(0)
data_iris = []
for i in range(sheet.nrows):
    row = sheet.row_values(i)
    data_iris.append(row[0:-1])
print('data iris', data_iris)
data_iris = array(data_iris)
print('data iris', data_iris)
print('data', data)
""" visualize """
# trans = pca(data, 3)[0]
trans_iris = pca(data_iris)[0]

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




ax8.scatter(trans_iris[:50, 0], trans_iris[:50, 1], c='r')
ax8.scatter(trans_iris[50:100, 0], trans_iris[50:100, 1], c='g')
ax8.scatter(trans_iris[100:, 0], trans_iris[100:, 1], c='b')
# ax1.scatter(data[:50, 0], data[:50, 1], c='r')
# ax1.scatter(data[50:, 0], data[50:, 1], c='b')
# ax2.scatter(trans[:50, 0], trans[:50, 1], c='r')
# ax2.scatter(trans[50:, 0], trans[50:, 1], c='b')
show()
