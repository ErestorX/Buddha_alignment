import os
import cv2
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def ldk_on_im(ldk, mean, max, trans, inv=False):
    tmp = np.asarray(ldk.T.tolist() + [list([1] * 68)])
    if inv:
        tmp = (tmp.T @ np.linalg.inv(trans)).T
    else:
        tmp = (tmp.T @ trans).T
    return (tmp[:3].T * max) + mean


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


def crop_pict(data, bbox, projection):
    int_bbox = bbox.astype(np.int32)
    rect_xy, width, height = int_bbox[:2], int_bbox[2] - int_bbox[0], int_bbox[3] - int_bbox[1]
    cropped_data = data[rect_xy[1]:rect_xy[1] + height, rect_xy[0]:rect_xy[0] + width, :]
    tmp = projection[:, :2] - int_bbox[:2]
    cropped_gt = np.zeros([68, 3])
    cropped_gt[:, :2] = tmp
    cropped_gt[:, 2] = projection[:, 2]
    return cropped_data, cropped_gt


class Artifact:
    def __init__(self, json_file, ds_path):
        json_data = json.load(json_file)
        self.id = json_data["artifact_id"]
        self.gt = np.asarray(json_data["avg_model"]) + np.asarray(json_data["hand_updates"])
        picture_ids = []
        self.pictures = []
        tmp = [os.path.basename(picture).split('\\')[-1] for picture in json_data['norm_preds_dict'].keys()]
        for file in tmp:
            if os.path.exists(os.path.join(ds_path, self.id, file)):
                picture_ids.append(file)
        keys = [f for f in json_data['norm_preds_dict'].keys()]
        for file, key in zip(picture_ids, keys):
            path2img = os.path.join(ds_path, self.id, file)
            self.pictures.append(Image(path2img, key, json_data))


class Image:
    def __init__(self, path2img, file_key, artifact_data):
        self.id = path2img.split("/")[-1]
        self.data = cv2.imread(path2img)
        pred, self.mean, self.max, self.bbox = artifact_data['norm_preds_dict'][file_key]
        pred, self.mean, self.bbox = np.asarray(pred), np.asarray(self.mean), np.asarray(self.bbox)
        self.transformation = get_transform(np.asarray(artifact_data['avg_model']), pred)
        self.precomputed_gt = ldk_on_im(
            np.asarray(artifact_data['avg_model']) + np.asarray(artifact_data['hand_updates']),
            self.mean, self.max, self.transformation, True)
        self.cropped_data, self.cropped_gt = crop_pict(self.data, self.bbox, self.precomputed_gt)
        # fig, ax = plt.subplots()
        # ax.imshow(self.data)
        # rect_xy, width, height = self.bbox[:2], self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]
        # title = "x1={},y1={}-x2={},y2={} - x0={},y0={}-W={},H={}".format(int(self.bbox[0]), int(self.bbox[1]), int(self.bbox[2]), int(self.bbox[3]), int(rect_xy[0]), int(rect_xy[1]), int(width), int(height))
        # rect = patches.Rectangle(rect_xy, width, height, linewidth=1, edgecolor='b', facecolor='none')
        # ax.add_patch(rect)
        # ax.set_title(title)
        # plt.savefig(os.path.join('dataset_tmp', self.id))


class BuddhaDataset:
    def __init__(self, config):
        self.config = config
        self.tmp_folder = 'dataset_tmp/'
        if not os.path.exists(self.tmp_folder):
            os.mkdir(self.tmp_folder)
        self.pickle_ds_name = os.path.join(self.tmp_folder, 'ds.pkl')
        self.artifacts = []

    def load(self):
        if not os.path.exists(self.pickle_ds_name) or self.config.reset_ds:
            self.write_ds(self.config.ds_path)
        with open(self.pickle_ds_name, 'rb') as pkl_file:
            self.artifacts = pickle.load(pkl_file)

    def get_datasets(self):
        data = []
        label = []
        for art in self.artifacts:
            label.append([art.id, art.gt])
            list_data = []
            for img in art.pictures:
                list_data.append([img.id, img.cropped_data])
            data.append([art.id, list_data])
        label = np.asarray(label)
        indexes = list(range(len(label)))
        ids_test = np.random.choice(len(indexes), size=int(len(label) * self.config.split_test_eval[0]), replace=False)
        ids_test = -np.sort(-ids_test)
        for id in ids_test:
            del indexes[id]
        ids_eval = np.random.choice(len(indexes), size=int(len(label) * self.config.split_test_eval[1]), replace=False)
        ids_eval = -np.sort(-ids_eval)
        for id in ids_eval:
            del indexes[id]
        test_ds = [[data[id] for id in ids_test], [label[id] for id in ids_test]]
        eval_ds = [[data[id] for id in ids_eval], [label[id] for id in ids_eval]]
        train_ds = [[data[id] for id in indexes], [label[id] for id in indexes]]
        return np.asarray(train_ds), np.asarray(test_ds), np.asarray(eval_ds)

    def write_ds(self, ds_path):
        print("INFO: Re generating the dataset")
        if os.path.exists(self.pickle_ds_name):
            os.remove(self.pickle_ds_name)
        artifact_json = [name for name in os.listdir(ds_path) if os.path.isfile(os.path.join(ds_path, name))]
        annotated_id = [name.split('.')[0] for name in artifact_json]
        for id in annotated_id:
            with open(os.path.join(ds_path, id + '.json')) as json_file:
                art = Artifact(json_file, ds_path)
                if self.config.remove_singleton:
                    if len(art.pictures) != 1:
                        self.artifacts.append(art)
                else:
                    self.artifacts.append(art)
        with open(self.pickle_ds_name, 'wb') as pkl_file:
            pickle.dump(self.artifacts, pkl_file)


class Config:
    def __init__(self, conf_file):
        with open(conf_file) as json_file:
            conf_dict = json.load(json_file)
            self.remove_singleton = conf_dict["remove_singleton"]
            self.ds_path = conf_dict["ds_path"]
            self.split_test_eval = conf_dict["split_test_eval"]
            self.save_intermediate = conf_dict["save_intermediate"]
            self.save_predict = conf_dict["save_predict"]
            self.save_eval = conf_dict["save_eval"]
            self.path_products = conf_dict["path_products"]
            self.reset_ds = conf_dict["reset_ds"]
            self.train, self.test, self.eval = conf_dict["train"], conf_dict["test"], conf_dict["eval"]
            base = 'logs/pipeline'
            i = 1
            while os.path.exists(self.path_products):
                self.path_products = base + str(i)
                i += 1
            if self.save_intermediate or self.save_predict or self.save_eval:
                os.mkdir(self.path_products)


if __name__ == '__main__':
    ds = BuddhaDataset(Config('conf.json'))
    ds.load()
    ds.get_datasets()
