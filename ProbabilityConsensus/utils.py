import json
import numpy as np
from icecream import ic

def get_transform(A, B):
    def best_fit_transform(A, B):
        assert A.shape == B.shape
        m = A.shape[1]
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:
            Vt[m - 1, :] *= -1
            R = np.dot(Vt.T, U.T)
        t = centroid_B.T - np.dot(R, centroid_A.T)
        T = np.identity(m + 1)
        T[:m, :m] = R
        T[:m, m] = t
        return T

    assert A.shape == B.shape
    m = A.shape[1]
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)
    T = best_fit_transform(src[:m, :].T, dst[:m, :].T)
    src = np.dot(T, src)
    T = best_fit_transform(A, src[:m, :].T)
    return T


def normalizeView(view):
    with open("../std_model.json") as f:
        data = json.load(f)
        std_model = np.asarray(data["std_model"])
    trans = get_transform(std_model, view)
    tmp = np.asarray(view.T.tolist() + [list([1] * 68)])
    x = (tmp.T @ trans).T[:3].T
    _mean = np.mean(x, axis=0)
    _x = x - _mean
    _max = _x.max()
    x = (_x / (2 * _max)) + .5
    return [trans, _mean, _max], x


def normalizeViews(view_list):
    normalized_views = []
    for view in view_list:
        _, norm_view = normalizeView(view)
        normalized_views.append(norm_view)
    return np.asarray(normalized_views)


def getArtifactStats(artifact):
    concat_net_pred = []
    concat_net_visu = []
    for key in artifact:
        concat_net_pred.append(artifact[key]["net_pred"])
        concat_net_visu.append(artifact[key]["net_visi"])
    concat_net_pred, concat_net_visu = np.asarray(concat_net_pred), np.asarray(concat_net_visu)
    concat_net_pred = normalizeViews(concat_net_pred)
    net_mean = np.mean(concat_net_pred, axis=0)
    list_net_cov = []
    concat_net_pred = concat_net_pred.swapaxes(0, 1)
    concat_net_pred = concat_net_pred.swapaxes(1, 2)
    for same_pt in concat_net_pred:
        list_net_cov.append(np.cov(same_pt))
    list_net_cov = np.asarray(list_net_cov)
    return net_mean, list_net_cov


def main():
    with open("../dataset.json") as f:
        dataset = json.load(f)

    stat_dataset = []
    for art in dataset.values():
        stat_dataset.append(getArtifactStats(art))
    return stat_dataset
