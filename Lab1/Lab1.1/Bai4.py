import cv2
import numpy as np
import matplotlib.pyplot as plt

A = np.array([1, 1, 1])
B = np.array([-1, 1, 1])
C = np.array([1, -1, 1])
D = np.array([-1, -1, 1])
E = np.array([1, 1, -1])
F = np.array([-1, 1, -1])
G = np.array([1, -1, -1])
H = np.array([-1, -1, -1])

camera = np.array([2, 3, 5])

Points = dict(zip("ABCDEFGH", [A, B, C, D, E, F, G, H]))

edges = ["AB", "CD", "EF", "GH", "AC", "BD", "EG", "FH", "AE", "CG", "BF", "DH"]
points = {k: v - camera for k, v in Points.items()}


def pinhole(v):
    x, y, z = v
    if z == 0:
        return np.array([float('inf'), float('inf')])
    return np.array([x / z, y / z])


def rotate(R, v):
    return np.dot(R, v)


def getRotX(angle):
    Rx = np.zeros((3, 3))
    Rx[0, 0] = 1
    Rx[1, 1] = np.cos(angle)
    Rx[1, 2] = -np.sin(angle)
    Rx[2, 1] = np.sin(angle)
    Rx[2, 2] = np.cos(angle)

    return Rx


def getRotY(angle):
    Ry = np.zeros((3, 3))
    Ry[0, 0] = np.cos(angle)
    Ry[0, 2] = -np.sin(angle)
    Ry[2, 0] = np.sin(angle)
    Ry[2, 2] = np.cos(angle)
    Ry[1, 1] = 1

    return Ry


def getRotZ(angle):
    Rz = np.zeros((3, 3))
    Rz[0, 0] = np.cos(angle)
    Rz[0, 1] = -np.sin(angle)
    Rz[1, 0] = np.sin(angle)
    Rz[1, 1] = np.cos(angle)
    Rz[2, 2] = 1

    return Rz


plt.figure(figsize=(10, 10))
angles = [15, 30, 45, 60]
for angle in angles:
    Rz = getRotZ(angle)

    ps = {k: rotate(Rz, p) for k, p in points.items()}
    uvs = {k: pinhole(p) for k, p in ps.items()}

    for a, b in edges:
        ua, va = uvs[a]
        ub, vb = uvs[b]
        plt.plot([ua, ub], [va, vb], "ko-")
    plt.pause(0.5)

plt.axis("equal")
plt.grid()

plt.show()