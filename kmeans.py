import sys
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import math
from copy import deepcopy

with open('cluster_data.txt') as f:
    lines = f.readlines()
    x1 = np.array([line.split()[1] for line in lines])
    x = x1.astype(np.float)
    y1 = np.array([line.split()[2] for line in lines])
    y = y1.astype(np.float)

plt.scatter(x, y)
plt.show()

m_t = np.empty((0, 1))
theta_t = np.empty((0, 1))

for theta in np.arange(0.0, 1.8, 0.1):

    m = 1
    cm = np.array([[x[0], y[0]]])
    cmc = np.array([1])

    for xi, yi in it.izip(x, y):

        min_dist = sys.maxint
        ci = np.empty((0, 2), float)
        for cmi in cm:
            distance = math.sqrt(((xi - cmi[0]) ** 2) + ((yi - cmi[1]) ** 2))

            if distance <= min_dist:
                min_dist = distance
                ci = cmi

        if min_dist > theta:
            m = m + 1
            cm = np.append(cm, np.array([[xi, yi]]), axis=0)
            cmc = np.append(cmc, [1])

        else:
            i = 0
            for c in cm:
                if np.array_equal(c, ci):
                    break
                i = i + 1
            new_x = (cmc[i] * cm[i][0] + xi) / (cmc[i] + 1)
            new_y = (cmc[i] * cm[i][1] + yi) / (cmc[i] + 1)
            cm[i] = np.array([new_x, new_y])

    m_t = np.append(m_t, [m])
    theta_t = np.append(theta_t, [theta])

plt.plot(theta_t, m_t)
plt.xlabel('Theta')
plt.ylabel('Number of Clusters')
plt.title('BSAS (Theta)')
plt.show()

# We now know value of k = 2
k = 2


def dist(a, b, ax1=1):
    return np.linalg.norm(a - b, axis=ax1)


X = np.array(list(zip(x, y)))
C_x = np.random.randint(0, np.max(X)-0.75, size=k)
C_y = np.random.randint(0, np.max(X)-0.75, size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)

plt.scatter(x, y, c='blue', s=7)
plt.scatter(C_x, C_y, marker='X', s=200, c='g')
plt.show()

C_old = np.zeros(C.shape)
clusters = np.zeros(len(X))
error = dist(C, C_old, None)
while error != 0:
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    C_old = deepcopy(C)
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)

    colors = ['r', 'g', 'b']
    fig, ax = plt.subplots()
    for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
    ax.scatter(C[:, 0], C[:, 1], marker='X', s=200, c='#050505')
    plt.show()
