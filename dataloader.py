import os
import json
import numpy as np
import matplotlib.pyplot as plt

from buddha_dataset import get_transform


def load_basic(path, id_fold):
    """
    create dataset in shape [train, test], with train and test being [inputs, labels], inputs being
    [art_id, [img_id, img]] and labels being [3d_model, [2d_model]].
    :param path:
    :param id_fold:
    :return: list [[[art_id, [img_id, img]], [3d_model, [2d_model]]], [[art_id, [img_id, img]], [3d_model, [2d_model]]]]
    """
    fold_path = './dataset/fold_' + id_fold
    with open(path + '/folds_info.json') as file:
        data = json.load(file)
    train_split, test_split = data['fold_' + id_fold]['train'], data['fold_' + id_fold]['test']
    with open(fold_path + '/gt.json') as file:
        data = json.load(file)
    gt_train, gt_test = data['train'], data['test']
    ds = [[[], []], [[], []]]
    for art_id in train_split:
        inputs = [art_id, []]
        gts = [gt_train[art_id]["gt3d"], []]
        imgs_path = os.listdir(fold_path + '/' + art_id)
        for path in imgs_path:
            img_id = (path.split("/")[-1]).split('.')[0]
            img = plt.imread(fold_path + '/' + art_id + '/' + path, )[:, :, :3]
            inputs[1].append([img_id, img])
            gts[1].append(gt_train[art_id]["images"][img_id])
        ds[0][0].append(inputs)
        ds[0][1].append(gts)
    for art_id in test_split:
        inputs = [art_id, []]
        gts = [gt_test[art_id]["gt3d"], []]
        imgs_path = os.listdir(fold_path + '/' + art_id)
        for path in imgs_path:
            img_id = (path.split("/")[-1]).split('.')[0]
            img = plt.imread(fold_path + '/' + art_id + '/' + path, )[:, :, :3]
            inputs[1].append([img_id, img])
            gts[1].append(gt_test[art_id]["images"][img_id])
        ds[1][0].append(inputs)
        ds[1][1].append(gts)

    return ds


def load_preprocessed(path, id_fold):
    """
    create dataset in shape [train, test], with train and test being [inputs, labels], inputs being
    [art_id, [img_id, 3d_pred]] and labels being [3d_model, [2d_model]].
    :param path:
    :param id_fold:
    :return: list [[[art_id, [img_id, 3d_pred]], [3d_model, [2d_model]]], [[art_id, [img_id, 3d_pred]], [3d_model, [2d_model]]]]
    """
    fold_path = path + '/fold_' + id_fold
    with open(path + '/folds_info.json') as file:
        data = json.load(file)
    train_split, test_split = data['fold_' + id_fold]['train'], data['fold_' + id_fold]['test']
    with open(fold_path + '/post3DDFA.json') as file:
        data = json.load(file)
    train_data, test_data = data['train'], data['test']
    with open(fold_path + '/gt.json') as file:
        data = json.load(file)
    gt_train, gt_test = data['train'], data['test']
    ds = [[[], []], [[], []]]
    for art_id in train_split:
        inputs = [art_id, []]
        gts = [gt_train[art_id]["gt3d"], []]
        imgs_path = os.listdir(fold_path + '/' + art_id)
        for path in imgs_path:
            img_id = (path.split("/")[-1]).split('.')[0]
            data = train_data[art_id][img_id]
            inputs[1].append([img_id, data])
            gts[1].append(gt_train[art_id]["images"][img_id])
        ds[0][0].append(inputs)
        ds[0][1].append(gts)
    for art_id in test_split:
        inputs = [art_id, []]
        gts = [gt_test[art_id]["gt3d"], []]
        imgs_path = os.listdir(fold_path + '/' + art_id)
        for path in imgs_path:
            img_id = (path.split("/")[-1]).split('.')[0]
            data = test_data[art_id][img_id]
            inputs[1].append([img_id, data])
            gts[1].append(gt_test[art_id]["images"][img_id])
        ds[1][0].append(inputs)
        ds[1][1].append(gts)
    return ds


def process_3d_pred(artifact, covariance=None):
    # normed_artifact = [[art_id, [img_id, 3d_pred]], [3d_model, [2d_model, transformation]]]
    # stat_art = [[art_id, cov_vect], [3d_model, [img_id, 2d_model, transformation]]]
    with open("./std_model.json") as f:
        data = json.load(f)
        std_model = np.asarray(data["std_model"])
    normed_artifact = [[artifact[0][0], []], [artifact[1][0], []]]
    for img_data, img_label in zip(artifact[0][1], artifact[1][1]):
        img = np.asarray(img_data[1])
        trans = get_transform(std_model, img)
        tmp = np.asarray(img.T.tolist() + [list([1] * 68)])
        x = (tmp.T @ trans).T[:3].T
        _mean = np.mean(x, axis=0)
        _x = x - _mean
        _max = _x.max()
        x = (_x / (2 * _max)) + .5
        transformation = [trans.tolist(), _mean.tolist(), _max]
        normed_artifact[0][1].append([img_data[0], x.tolist()])
        normed_artifact[1][1].append([img_label, transformation])

    if covariance is None:
        return normed_artifact

    vals = []
    for pred in normed_artifact[0][1]:
        vals.append(pred[1])
    vals = np.asarray(vals)
    mean = np.mean(vals, axis=0).flatten()

    if covariance is 'full':
        vect_stack = []
        for pred in normed_artifact[0][1]:
            vect_stack.append(np.asarray(pred[1]).flatten())
        vect_stack = np.asarray(vect_stack)
        cov_vect = np.cov(vect_stack.swapaxes(0, 1)).flatten()

    if covariance is 'dimension':
        vect_stack = []
        for pred in normed_artifact[0][1]:
            vect_stack.append(pred[1])
        vect_stack = np.asarray(vect_stack)
        vect_stack = vect_stack.swapaxes(0, 2)
        cov_stack = []
        for dim in vect_stack:
            cov_stack.append(np.cov(dim))
        cov_vect = np.asarray(cov_stack).flatten()

    if covariance is 'point':
        vect_stack = []
        for pred in normed_artifact[0][1]:
            vect_stack.append(np.asarray(pred[1]))
        vect_stack = np.asarray(vect_stack)
        vect_stack = vect_stack.swapaxes(0, 1)
        vect_stack = vect_stack.swapaxes(1, 2)
        cov_stack = []
        for point in vect_stack:
            cov_stack.append(np.cov(point))
        cov_vect = np.asarray(cov_stack).flatten()

    stat_art = [[normed_artifact[0][0], np.concatenate([mean, cov_vect])], [normed_artifact[1][0], []]]
    for img_data, img_label in zip(normed_artifact[0][1], normed_artifact[1][1]):
        new_img_label = [img_data[0], img_data[1], img_label]
        stat_art[1][1].append(new_img_label)
    return stat_art

if __name__ == '__main__':
    path = './dataset'
    # ds = load_basic(path, '1')
    ds_preprocessed = load_preprocessed(path, '1')
    # prep_full = process_3d_pred([ds_preprocessed[0][0], ds_preprocessed[0][1]], covariance='full')
    # prep_dim = process_3d_pred(ds_preprocessed[0][0], covariance='dimension')
    prep_pt = process_3d_pred([ds_preprocessed[0][0][0], ds_preprocessed[0][1][0]], covariance='point')
    print(prep_pt)
