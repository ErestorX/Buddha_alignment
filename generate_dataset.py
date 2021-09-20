import json
import cv2
from icecream import ic
import os.path
import pathlib
import pickle
import numpy as np
import matplotlib.pyplot as plt
import copy
import yaml
from TDDFA_ONNX import TDDFA_ONNX
from buddha_dataset import BuddhaDataset, Artifact, Image, Config, crop_pict, get_transform


def generate_folds_ids(folds=5):
    file_path = './dataset.json'
    with open(file_path) as file:
        data = json.load(file)

    all_artifacts = list(data.keys())
    nb_artifacts = len(all_artifacts)
    train_split_index = nb_artifacts//folds
    all_artifacts = np.asarray(all_artifacts)
    np.random.shuffle(all_artifacts)

    folds_file = './dataset/folds_info.json'
    folds_dict = {}
    with open(folds_file, 'w+') as file:
        for id_fold in range(folds):
            fold_key = 'fold_' + str(id_fold + 1)
            try:
                os.mkdir('./dataset/' + fold_key)
            except:
                pass
            folds_dict[fold_key] = list(all_artifacts[train_split_index*id_fold:train_split_index*(id_fold+1)])
        json.dump(folds_dict, file)


def data_augment(x, labels_2d):
    keep_crop_ratio = 0.9
    x_dim = np.asarray(x.shape[:2])
    crop_dim = (x_dim * keep_crop_ratio).astype(np.int)
    max_disp_width, max_disp_height = max(1, x_dim[0] * (1 - keep_crop_ratio)), max(1, x_dim[1] * (1 - keep_crop_ratio))
    displacement = [np.random.randint(0, max_disp_width), np.random.randint(0, max_disp_height)]
    x = x[displacement[0]: displacement[0] + crop_dim[0], displacement[1]: displacement[1] + crop_dim[1]]
    for id, val in enumerate(labels_2d):
        if not np.array_equal(val, [0, 0]):
            labels_2d[id][0] = val[0] - displacement[1]
            labels_2d[id][1] = val[1] - displacement[0]
    return x, labels_2d


def process_art(fold_id, art, augment=True):
    art_folder = './dataset/' + fold_id + '/' + art.id
    if not os.path.isdir(art_folder):
        os.mkdir(art_folder)
    art_gts = []
    for picture in art.pictures:
        file_2d_annot = 'dataset_2d/' + art.id.split('_')[0] + '_' + art.id.split('_')[1] + '_' + picture.id.split('.')[0] + '_2D.json'
        try:
            with open(file_2d_annot) as file:
                data_2d = json.load(file)
        except:
            continue
        gt_2d = data_2d['landmarks']
        for id, val in enumerate(gt_2d):
            if np.array_equal(val, [0]):
                gt_2d[id] = [0, 0]
        img, bbox = picture.data, picture.bbox
        tmp = np.zeros([68, 3])
        tmp[:, :2] = np.asarray(gt_2d)
        if not augment:
            original_crop_img, original_crop_gt_2d = crop_pict(img, bbox.astype(np.int32), tmp)
            plt.imsave(art_folder + '/' + picture.id.split('.')[0] + '.png', original_crop_img)
            art_gts.append([picture.id.split('.')[0], original_crop_gt_2d])
        else:
            int_bbox = bbox.astype(np.int32)
            width, height = int_bbox[2] - int_bbox[0], int_bbox[3] - int_bbox[1]
            half_width_inc, half_height_inc = (width*.15).astype(np.int32), (height*.15).astype(np.int32)
            int_bbox[0], int_bbox[1], int_bbox[2], int_bbox[3] = int_bbox[0] - half_width_inc, int_bbox[1] - half_height_inc, int_bbox[2] + half_width_inc, int_bbox[3] + half_height_inc
            int_bbox[0], int_bbox[1] = max(int_bbox[0], 0), max(int_bbox[1], 0)
            int_bbox[3], int_bbox[2] = min(int_bbox[3], img.shape[0]), min(int_bbox[2], img.shape[1])
            crop_img, crop_gt_2d = crop_pict(img, int_bbox, tmp)
            crop_gt_2d = crop_gt_2d[:, :2]
            aug_img, aug_gt_2d = data_augment(crop_img, crop_gt_2d)
            for id, val in enumerate(gt_2d):
                if np.array_equal(val, [0, 0]):
                    aug_gt_2d[id] = [0, 0]
            plt.imsave(art_folder + '/' + picture.id.split('.')[0] + '.png', aug_img)
            art_gts.append([picture.id.split('.')[0], aug_gt_2d])
    return art_gts


def normalize_position(input):
    with open("./std_model.json") as f:
        data = json.load(f)
        std_model = np.asarray(data["std_model"])
    trans = get_transform(std_model, input)
    tmp = np.asarray(input.T.tolist() + [list([1] * 68)])
    x = (tmp.T @ trans).T[:3].T
    _mean = np.mean(x, axis=0)
    _x = x - _mean
    _max = _x.max()
    x = (_x / (2 * _max)) + .5
    return [trans, _mean, _max], x


def make_cropped_and_3d_gt(nb_aug=5):
    with open('./dataset/folds_info.json') as file:
        folds = json.load(file)
    fold_ids = list(folds.keys())

    ds = BuddhaDataset(Config('conf_tmp.json'))
    ds.load()
    ds = ds.artifacts

    for fold_id in fold_ids:
        fold_json = {}
        for art in ds:
            if art.id in folds[fold_id]:
                gt3d = art.gt
                _, gt3d = normalize_position(gt3d)
                tmp_art = copy.deepcopy(art)
                tmp_art.id = art.id + '_aug0'
                fold_json[tmp_art.id] = {'gt3d': gt3d.tolist(), 'images': {}}
                img_gt2d = process_art(fold_id, tmp_art, augment=False)
                for val in img_gt2d:
                    fold_json[tmp_art.id]['images'][val[0]] = val[1].tolist()
                for i in range(1, nb_aug):
                    tmp_art.id = art.id + '_aug' + str(i)
                    fold_json[tmp_art.id] = {'gt3d': gt3d.tolist(), 'images': {}}
                    img_gt2d = process_art(fold_id, tmp_art)
                    for val in img_gt2d:
                        fold_json[tmp_art.id]['images'][val[0]] = val[1].tolist()
        with open('./dataset/' + fold_id + '/gt.json', 'w+') as file:
            json.dump(fold_json, file)


def make_intermediate_ds(nb_aug=5):
    with open('./dataset/folds_info.json') as file:
        folds = json.load(file)
    fold_ids = list(folds.keys())
    cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'
    tddfa = TDDFA_ONNX(**cfg)
    for fold_id in fold_ids:
        prediction_ds = {}
        for art in folds[fold_id]:
            for i in range(nb_aug):
                prediction_ds[art + '_aug' + str(i)] = {}
                imgs_path = os.listdir('./dataset/' + fold_id + '/' + art + '_aug' + str(i))
                for path in imgs_path:
                    img_id = (path.split("/")[-1]).split('.')[0]
                    img = cv2.imread('./dataset/' + fold_id + '/' + art + '_aug' + str(i) + '/' + path, )
                    face_bbox = [0, 0, img.shape[0], img.shape[1]]
                    param_lst, roi_box_lst = tddfa(img, [face_bbox])
                    pred = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
                    pred = pred[0].T
                    prediction_ds[art + '_aug' + str(i)][img_id] = pred.tolist()
        with open('./dataset/' + fold_id + '/post3DDFA.json', 'w+') as file:
            json.dump(prediction_ds, file)


if __name__ == '__main__':
    nb_aug = 5
    generate_folds_ids()
    make_cropped_and_3d_gt(nb_aug)
    make_intermediate_ds(nb_aug)




