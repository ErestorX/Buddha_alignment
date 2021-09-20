import numpy as np


def umeyama(P, Q):
    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(np.transpose(centeredP), centeredQ) / n

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    Rot = np.dot(V, W)

    varP = np.var(P, axis=0).sum()
    scale = 1 / varP * np.sum(S)  # scale factor

    trans = Q.mean(axis=0) - P.mean(axis=0).dot(scale * Rot)

    return scale, Rot, trans


ldk = np.random.random([68, 3])
std_model = np.random.random([68, 3])

scale, Rot, trans = umeyama(ldk, std_model)
x = ldk.dot(scale * Rot) + trans
x = (x - trans).dot(np.linalg.inv(scale * Rot))

for y, x in zip(ldk, x):
    print(np.sqrt((y[0]-x[0])**2 + (y[1]-x[1])**2 + (y[2]-x[2])**2))