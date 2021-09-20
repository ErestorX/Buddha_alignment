import os
import cv2
import json
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from mpl_toolkits.mplot3d import Axes3D

from buddha_dataset import get_transform


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


def get_visible_landmarks(ldks):
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
    triangles = [ldks[triangle] for triangle in np.asarray(triangles)]
    is_visible_args = [[ldk, triangles] for ldk in ldks]
    with Pool() as pool:
        visibility = pool.map(is_visible, is_visible_args)
    return visibility


def is_visible(is_visible_args):
    point, triangles = is_visible_args
    triangles_with_point, triangles_without_point = [], []
    for tmp in triangles:
        if point in tmp:
            triangles_with_point.append(tmp)
        else:
            triangles_without_point.append(tmp)
    # find one obscuring triangle in the mesh
    for triangle in triangles_without_point:
        # get orthogonal projection of point on the triangle plane
        plane_norm = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        ndotu = plane_norm.dot(np.array([0, 0, 1]))
        w = point - triangle[0]
        pt_on_tri = w + (-plane_norm.dot(w) / ndotu) * np.array([0, 0, 1]) + triangle[0]
        if point[2] > pt_on_tri[2]:
            continue
        # find if the projected point is within the triangle perimeter
        area = np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
        area_0 = np.linalg.norm(np.cross(triangle[1] - pt_on_tri, triangle[2] - pt_on_tri))
        area_1 = np.linalg.norm(np.cross(triangle[2] - pt_on_tri, triangle[0] - pt_on_tri))
        area_2 = np.linalg.norm(np.cross(triangle[0] - pt_on_tri, triangle[1] - pt_on_tri))
        area_diff = np.abs(area - np.sum([area_0, area_1, area_2]))
        # if the sum of areas of the small triangles is different from the original area, with depth approx.
        if area_diff < 1e-3:
            return False
    # find at least one non-back-facing triangle
    for triangle in triangles_with_point:
        norm = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        if np.arccos(norm[-1] / np.linalg.norm(norm)) > np.pi / 2:
            return True
    return False


def print_cloud_pt(art):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cloud_0 = art['imgs'][0]['img_ldk']
    cloud_1 = art['imgs'][1]['img_ldk']
    ax.scatter(cloud_0[:, 0], cloud_0[:, 1], cloud_0[:, 2])
    plt.savefig("prev_tmp_fig0.png")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(cloud_1[:, 0], cloud_1[:, 1], cloud_1[:, 2])
    plt.savefig("prev_tmp_fig1.png")


def load_basic(path, test_fold, nb_augment=5):
    """
    create dataset in shape [train, test], with train and test being [inputs, labels], inputs being
    [art_id, [img_id, img]] and labels being [3d_model, [2d_model]].
    :param nb_augment:
    :param path:
    :param test_fold:
    :return: list [[[art_id, [img_id, img]], [3d_model, [2d_model]]], [[art_id, [img_id, img]], [3d_model, [2d_model]]]]
    """
    with open(path + '/folds_info.json') as file:
        data = json.load(file)
    ds = [[], []]
    for fold in data.keys():
        fold_path = path + '/' + fold
        with open(fold_path + '/gt.json') as file:
            fold_gt = json.load(file)
        fold_ds = []
        for art_id in data[fold]:
            tmp_nb_augment = 1 if fold == test_fold else nb_augment
            for i in range(tmp_nb_augment):
                artifact = {'art_id': art_id + '_aug' + str(i),
                            'art_gt': np.asarray(fold_gt[art_id + '_aug' + str(i)]["gt3d"]),
                            'imgs': []}
                imgs_path = os.listdir(fold_path + '/' + art_id + '_aug' + str(i))
                for x in imgs_path:
                    image = {'img_id': (x.split("/")[-1]).split('.')[0],
                             'img_gt': np.asarray(
                                 fold_gt[art_id + '_aug' + str(i)]["images"][(x.split("/")[-1]).split('.')[0]]),
                             'img_cv2': cv2.imread(fold_path + '/' + art_id + '_aug' + str(i) + '/' + x, )}
                    artifact['imgs'].append(image)
                fold_ds.append(artifact)
        if fold == test_fold:
            ds[1] += fold_ds
        else:
            ds[0] += fold_ds
    return ds


def load_preprocessed(path, test_fold, nb_augment=5):
    """
    create dataset in shape [train, test], with train and test being [inputs, labels], inputs being
    [art_id, [img_id, 3d_pred]] and labels being [3d_model, [2d_model]].
    :param nb_augment:
    :param test_fold:
    :param path:
    :return: list [[[art_id, [img_id, 3d_pred]], [3d_model, [2d_model]]], [[art_id, [img_id, 3d_pred]], [3d_model, [2d_model]]]]
    """
    with open(path + '/folds_info.json') as file:
        data = json.load(file)
    ds = [[], []]
    for fold in data.keys():
        fold_path = path + '/' + fold
        with open(fold_path + '/gt.json') as file:
            fold_gt = json.load(file)
        with open(fold_path + '/post3DDFA.json') as file:
            preprocessed_input = json.load(file)
        fold_ds = []
        for art_id in data[fold]:
            tmp_nb_augment = 1 if fold == test_fold else nb_augment
            for i in range(tmp_nb_augment):
                artifact = {'art_id': art_id + '_aug' + str(i),
                            'art_gt': np.asarray(fold_gt[art_id + '_aug' + str(i)]["gt3d"]),
                            'imgs': []}
                imgs_path = os.listdir(fold_path + '/' + art_id + '_aug' + str(i))
                for x in imgs_path:
                    image = {'img_id': (x.split("/")[-1]).split('.')[0],
                             'img_gt': np.asarray(
                                 fold_gt[art_id + '_aug' + str(i)]["images"][(x.split("/")[-1]).split('.')[0]]),
                             'img_ldk': preprocessed_input[art_id + '_aug' + str(i)][(x.split("/")[-1]).split('.')[0]]}
                    artifact['imgs'].append(image)
                fold_ds.append(artifact)
        if fold == test_fold:
            ds[1] += fold_ds
        else:
            ds[0] += fold_ds
    return ds


def process_3d_pred(artifact, covariance=None, raw_art=None):
    # artifact{'art_id': str, 'art_gt': np.array, 'imgs': list[image]}
    # image{'img_id': str, 'img_gt': np.array, 'img_ldk': np.array}
    with open("./std_model.json") as f:
        data = json.load(f)
        std_model = np.asarray(data["std_model"])
    for image_index in range(len(artifact['imgs'])):
        image = artifact['imgs'][image_index]
        ldk = np.asarray(image['img_ldk'])
        # trans = get_transform(std_model, ldk)
        # tmp = np.asarray(ldk.T.tolist() + [list([1] * 68)])
        # x = (tmp.T @ trans).T[:3].T
        # _mean = np.mean(x, axis=0)
        # _x = x - _mean
        # _max = _x.max()
        # x = (_x / (2 * _max)) + .5
        # transformation = [trans.tolist(), _mean.tolist(), _max]
        transformation = umeyama(ldk, std_model)
        x = ldk.dot(transformation[0] * transformation[1]) + transformation[2]
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(x[:, 0], x[:, 1], x[:, 2], c='r')
        # ax.scatter(new_x[:, 0], new_x[:, 1], new_x[:, 2], c='b')
        # ax.set_title('red prev, blue new, error ' + str(((x - new_x) ** 2).sum()))
        # plt.savefig('./logs/visu_norm_methods/'+artifact['art_id'] + '_' + image['img_id'])
        image['img_rot'] = transformation
        image['img_ldk_std'] = x
        image['img_ldk_vis'] = get_visible_landmarks(np.asarray(image['img_ldk']))
        if raw_art is not None:
            image['img_cv2'] = raw_art['imgs'][image_index]['img_cv2']

    if covariance is None:
        # return artifact with all point clouds oriented and scaled
        return artifact

    vals = []
    for image in artifact['imgs']:
        vals.append(image['img_ldk_std'])
    vals = np.asarray(vals)
    mean = np.mean(vals, axis=0).flatten()

    if covariance is 'full':
        # return artifact with point cloud summed in 68 points average plus 204*204 covariance matrix
        vect_stack = []
        for image in artifact['imgs']:
            vect_stack.append(np.asarray(image['img_ldk_std']).flatten())
        vect_stack = np.asarray(vect_stack)
        cov_vect = np.cov(vect_stack.swapaxes(0, 1)).flatten()

    if covariance is 'dimension':
        # return artifact with point cloud summed in 68 points average plus 3*68*68 covariance matrix
        vect_stack = []
        for image in artifact['imgs']:
            vect_stack.append(image['img_ldk_std'])
        vect_stack = np.asarray(vect_stack)
        vect_stack = vect_stack.swapaxes(0, 2)
        cov_stack = []
        for dim in vect_stack:
            cov_stack.append(np.cov(dim))
        cov_vect = np.asarray(cov_stack).flatten()

    if covariance is 'point':
        # return artifact with point cloud summed in 68 points average plus 68*3*3 covariance matrix
        vect_stack = []
        for image in artifact['imgs']:
            vect_stack.append(np.asarray(image['img_ldk_std']))
        vect_stack = np.asarray(vect_stack)
        vect_stack = vect_stack.swapaxes(0, 1)
        vect_stack = vect_stack.swapaxes(1, 2)
        cov_stack = []
        for point in vect_stack:
            cov_stack.append(np.cov(point))
        cov_vect = np.asarray(cov_stack).flatten()

    artifact['imgs_cov'] = np.concatenate([mean, cov_vect])
    return artifact


def print_valid(ds_img, ds_preprocessed):
    path = 'logs/visu_3DDFA/'
    ds_train, ds_test = ds_img
    ds_preprocessed_train, ds_preprocessed_test = ds_preprocessed
    for art_img, art_processed in zip(ds_train, ds_preprocessed_train):
        images = art_img['imgs']
        landmarks = art_processed['imgs']
        for img, ldk in zip(images, landmarks):
            fig, ax = plt.subplots()
            ax.imshow(img['img'])
            points = np.asarray(ldk['img_ldk'])
            ax.scatter(points[:, 0], points[:, 1], c="r", s=15)
            points = img['img_gt']
            ax.scatter(points[:, 0], points[:, 1], c="b", s=15)
            plt.savefig(os.path.join(path, art_img['art_id'] + '_' + img['img_id']))
            plt.close(fig)
        break


if __name__ == '__main__':
    path = './dataset'
    ds = load_basic(path, 'fold_1')
    ds_preprocessed = load_preprocessed(path, 'fold_1')
    # ds_preprocessed[train/test][artifact]
    # artifact{'art_id': str, 'art_gt': np.array, 'imgs': list[image]}
    # image{'img_id': str, 'img_gt': np.array, 'img_ldk': np.array}
    # for artifact_index in range(len(ds_preprocessed[0])):
    start = time.time()
    for artifact_index in range(len(ds_preprocessed[0])):
        if artifact_index % 100 == 0:
            end = time.time()
            print(artifact_index, 'out of', len(ds_preprocessed[0]), 'in', end - start)
            start = time.time()
        # covariance in ['full', 'dimension', 'point', None]
        ds_preprocessed[0][artifact_index] = process_3d_pred(ds_preprocessed[0][artifact_index], covariance=None,
                                                             raw_art=ds[0][artifact_index])
    for artifact_index in range(len(ds_preprocessed[1])):
        if artifact_index % 100 == 0:
            end = time.time()
            print(artifact_index, 'out of', len(ds_preprocessed[1]), 'in', end - start)
            start = time.time()
        # covariance in ['full', 'dimension', 'point', None]
        ds_preprocessed[1][artifact_index] = process_3d_pred(ds_preprocessed[1][artifact_index], covariance=None,
                                                             raw_art=ds[1][artifact_index])
    with open('ds_full.pkl', 'wb') as file:
        pickle.dump(ds_preprocessed, file)
    # print_cloud_pt(prep_art)
    # print_valid(ds, ds_preprocessed)
