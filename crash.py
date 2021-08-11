import json
import matplotlib.pyplot as plt
import numpy as np
from buddha_dataset import BuddhaDataset, Artifact, Image, Config, get_transform, ldk_on_im


def _get_attention(input):
    x = input
    triangles = [[0, 1, 36], [1, 48, 36], [1, 2, 48], [2, 3, 48], [3, 4, 48], [4, 60, 48], [4, 5, 60], [5, 59, 60],
                 [5, 6, 59], [6, 58, 59], [6, 7, 58], [7, 57, 58], [7, 8, 57], [8, 9, 57], [9, 56, 57], [9, 10, 56],
                 [10, 55, 56], [10, 11, 55], [11, 64, 55], [11, 12, 64], [12, 64, 54], [12, 13, 54], [13, 14, 54],
                 [14, 15, 54], [15, 45, 54], [15, 16, 45], [16, 26, 45], [26, 25, 45], [25, 44, 45], [25, 24, 44],
                 [24, 43, 44], [23, 43, 24], [23, 42, 43], [22, 42, 23], [21, 22, 23], [20, 21, 23], [20, 39, 21],
                 [20, 38, 39], [19, 38, 20], [19, 37, 38], [18, 37, 19], [18, 36, 37], [17, 36, 18], [0, 36, 17],
                 [36, 41, 37], [36, 41, 40], [40, 38, 37], [38, 40, 39], [42, 47, 43], [43, 47, 44], [44, 47, 46],
                 [44, 46, 45], [21, 39, 27], [27, 39, 28], [28, 39, 29], [29, 39, 31], [39, 40, 31], [40, 41, 31],
                 [31, 41, 36], [31, 36, 48], [21, 27, 22], [22, 27, 42], [27, 28, 42], [28, 29, 42], [29, 35, 42],
                 [35, 47, 42], [35, 46, 47], [35, 45, 46], [35, 54, 45], [29, 31, 30], [30, 31, 32], [30, 32, 33],
                 [30, 33, 34], [30, 34, 35], [29, 30, 35], [31, 48, 49], [31, 49, 50], [31, 50, 32], [32, 50, 33],
                 [33, 50, 51], [33, 51, 52], [33, 52, 34], [34, 52, 35], [35, 52, 53], [35, 53, 54], [48, 60, 49],
                 [49, 61, 50], [50, 61, 51], [51, 61, 62], [51, 62, 63], [51, 63, 52], [52, 63, 53], [53, 64, 54],
                 [49, 60, 59], [49, 59, 61], [49, 67, 61], [61, 67, 62], [62, 67, 66], [62, 66, 65], [62, 65, 63],
                 [55, 63, 65], [53, 65, 55], [53, 55, 64], [58, 67, 59], [58, 66, 67], [57, 66, 58], [56, 66, 57],
                 [56, 65, 66], [55, 65, 56]]
    triangles = [x[triangle] for triangle in np.asarray(triangles)]
    attention = _are_visible(x, triangles)
    return attention

def _are_visible(list_x, triangles):
    result = []
    for x in list_x:
        visible = True
        for triangle in triangles:
            if x in triangle:
                continue
            pt = _intersect(x, triangle)
            if x[2] > pt[2]:
                continue
            if _is_within(pt, triangle):
                visible = False
                result.append(False)
                break
        if visible:
            result.append(_not_back_facing(x, triangles))
    return np.asarray(result)

def _intersect(x, triangle):
    planePoint = triangle[0]
    planeNormal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    rayPoint = x
    rayDirection = np.array([0, 0, 1])
    ndotu = planeNormal.dot(rayDirection)
    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    return w + si * rayDirection + planePoint

def _is_within(pt, triangle):
    area = 0.5 * np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
    sub_0 = 0.5 * np.linalg.norm(np.cross(triangle[1] - pt, triangle[2] - pt))
    sub_1 = 0.5 * np.linalg.norm(np.cross(triangle[2] - pt, triangle[0] - pt))
    sub_2 = 0.5 * np.linalg.norm(np.cross(triangle[0] - pt, triangle[1] - pt))
    x = np.abs(area - np.sum([sub_0, sub_1, sub_2]))
    return x < 1e-3

def _not_back_facing(x, triangles):
    list_triangles = [triangle for triangle in triangles if x in triangle]
    for triangle in list_triangles:
        norm = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        angle = 180 * np.arccos(norm[-1] / np.linalg.norm(norm)) / np.pi
        if angle > 90:
            return True
    return False


conf = Config('conf.json')
ds = BuddhaDataset(conf)
ds.load()

json_data = {}
for id, art in enumerate(ds.artifacts):
    print("artifact", id, "out of", len(ds.artifacts))
    json_data[art.id] = {}
    for img in art.pictures:
        net_visibility = _get_attention(img.pred)
        gt_visibility = _get_attention(img.precomputed_gt)
        json_data[art.id][img.id] = {"net_pred": img.pred.tolist(), "net_visi": net_visibility.tolist(),
                                     "gt": img.precomputed_gt.tolist(), "gt_visi": gt_visibility.tolist()}

with open("dataset.json", "w+") as file:
    json.dump(json_data, file)
