import os
import cv2
import numpy as np
import random
import json
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def revert_norm(points, mean, max):
    return (points * max) + mean


def ldk_on_im(ldk, mean, max, trans, inv=False):
    tmp = np.asarray(ldk.T.tolist() + [list([1] * 68)])
    if inv:
        tmp = (tmp.T @ np.linalg.inv(trans)).T
    else:
        tmp = (tmp.T @ trans).T
    return revert_norm(tmp[:3].T, mean, max)


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
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)
    T = best_fit_transform(src[:m,:].T, dst[:m,:].T)
    src = np.dot(T, src)
    T = best_fit_transform(A, src[:m,:].T)
    return T


def write_ds(ds_path):
    print("INFO: Re generating the dataset")
    artifact_json = [name for name in os.listdir(ds_path) if os.path.isfile(os.path.join(ds_path, name))]
    annotated_id = [name.split('.')[0] for name in artifact_json]
    ds = dict.fromkeys(annotated_id)
    for id in annotated_id:
        with open(os.path.join(ds_path, id + '.json')) as json_file:
            annotation = json.load(json_file)
            pictures = []
            tmp = [os.path.basename(picture).split('\\')[-1] for picture in annotation['norm_preds_dict'].keys()]
            for file in tmp:
                if os.path.exists(os.path.join(ds_path, id, file)):
                    pictures.append(file)
            saved_filenames = [f for f in annotation['norm_preds_dict'].keys()]
            ds[id] = {"machine_gt": np.asarray(annotation['avg_model']),
                      "human_gt": np.asarray(annotation['hand_updates']),
                      "pictures": dict.fromkeys(pictures)}

            for file, saved_filename in zip(pictures, saved_filenames):
                data = cv2.imread(os.path.join(ds_path, id, file))
                if data is None:
                    print(file)
                pred, mean, max, bbox = annotation['norm_preds_dict'][saved_filename]
                pred, mean, bbox = np.asarray(pred), np.asarray(mean), np.asarray(bbox)
                T = get_transform(np.asarray(annotation['avg_model']), pred)
                projection = ldk_on_im(np.asarray(annotation['avg_model']) + np.asarray(annotation['hand_updates']),
                                       mean, max, T, True)
                ds[id]['pictures'][file] = {"data": data, "transformation": T, "mean": mean,
                                            "max": max, "bbox": bbox, "precomputed_gt": projection}
    with open('Buddha_ds.pkl', 'wb') as pkl_file:
        pickle.dump(ds, pkl_file)


def load_ds(ds_path, remove_singleton=True):
    artifact_json = [name for name in os.listdir(ds_path) if os.path.isfile(os.path.join(ds_path, name))]
    annotated_id = [name.split('.')[0] for name in artifact_json]
    if not os.path.exists('Buddha_ds.pkl'):
        write_ds(ds_path)
    with open('Buddha_ds.pkl', 'rb') as pkl_file:
        ds = pickle.load(pkl_file)
    ds_keys = ds.keys()
    for id in annotated_id:
        if id not in ds_keys:
            write_ds(ds_path)
            with open('Buddha_ds.pkl', 'rb') as pkl_file:
                ds = pickle.load(pkl_file)
            break
    if remove_singleton:
        print("Artifact count:", len(ds))
        for key in list(ds.keys()):
            if len(ds[key]['pictures']) == 1:
                ds.pop(key)
        print("Artifact count without singleton:", len(ds))
    return ds


def visualize_artifact(artifact_id, artifact):
    save_folder = os.path.join("examples/results", artifact_id)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for id in artifact['pictures'].keys():
        picture, landmarks, bbox = artifact['pictures'][id]['data'], artifact['pictures'][id]['precomputed_gt'][:, :2], artifact['pictures'][id]['bbox']
        fig, ax = plt.subplots()
        if picture is None:
            print(id)
            continue
        ax.imshow(picture)
        rect_xy, width, height = bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1]
        rect = patches.Rectangle(rect_xy, width, height, linewidth=1, edgecolor='b', facecolor='none')
        poly = patches.Polygon(landmarks, linewidth=.5, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.add_patch(poly)
        plt.savefig(os.path.join(save_folder, id))


if __name__ == '__main__':
    ds = load_ds('data')
    keys = random.sample(list(ds.keys()), 10)
    for key in keys:
        visualize_artifact(key, ds[key])
