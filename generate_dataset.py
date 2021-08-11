import json
from icecream import ic
import os.path
import pathlib
import pickle
import numpy as np
import matplotlib.pyplot as plt
import yaml
from TDDFA_ONNX import TDDFA_ONNX
from buddha_dataset import BuddhaDataset, Artifact, Image, Config, crop_pict, get_transform

def generate_folds_ids():
    train_split = 0.7
    folds = 5
    file_path = './dataset.json'
    with open(file_path) as file:
        data = json.load(file)

    all_artifacts = list(data.keys())
    nb_artifacts = len(all_artifacts)
    train_split_index = int(nb_artifacts * train_split)
    all_artifacts = np.asarray(all_artifacts)
    np.random.shuffle(all_artifacts)

    folds_file = './dataset/folds_info.json'
    folds_dict = {'fold_1': {"train": [], "test": []},
                  'fold_2': {"train": [], "test": []},
                  'fold_3': {"train": [], "test": []},
                  'fold_4': {"train": [], "test": []},
                  'fold_5': {"train": [], "test": []}}
    with open(folds_file, 'w+') as file:
        for id_fold in range(folds):
            fold_key = 'fold_' + str(id_fold + 1)
            folds_dict[fold_key]['train'] = list(all_artifacts[:train_split_index])
            folds_dict[fold_key]['test'] = list(all_artifacts[train_split_index:])
            np.random.shuffle(all_artifacts)

        json.dump(folds_dict, file)


def data_augment(x, labels_2d):
    proba_fliplr = 0.2
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
    if np.random.random([1]) < proba_fliplr:
        half_x = x.shape[1] // 2
        x = np.fliplr(x)
        for id, val in enumerate(labels_2d):
            if not np.array_equal(val, [0, 0]):
                labels_2d[id][0] = val[0] + 2*(half_x - val[0])
    return x, labels_2d


def process_art(fold_id, art):
    art_folder = './dataset/' + fold_id + '/' + art.id
    if not os.path.isdir(art_folder):
        os.mkdir(art_folder)
    art_gts = []
    for picture in art.pictures:
        file_2d_annot = 'dataset_2d/' + art.id + '_' + picture.id.split('.')[0] + '_2D.json'
        # with open(file_2d_annot) as file:
        #     data_2d = json.load(file)
        # gt_2d = data_2d['landmarks']
        # for id, val in enumerate(gt_2d):
        #     if np.array_equal(val, [0]):
        #         gt_2d[id] = [0, 0]
        img, bbox = picture.data, picture.bbox
        # plt.imsave(art_folder + '/' + picture.id.split('.')[0] + '_original.png', img)
        int_bbox = bbox.astype(np.int32)
        width, height = int_bbox[2] - int_bbox[0], int_bbox[3] - int_bbox[1]
        half_width_inc, half_height_inc = (width*.15).astype(np.int32), (height*.15).astype(np.int32)
        int_bbox[0], int_bbox[1], int_bbox[2], int_bbox[3] = int_bbox[0] - half_width_inc, int_bbox[1] - half_height_inc, int_bbox[2] + half_width_inc, int_bbox[3] + half_height_inc
        int_bbox[0], int_bbox[1] = max(int_bbox[0], 0), max(int_bbox[1], 0)
        int_bbox[3], int_bbox[2] = min(int_bbox[3], img.shape[0]), min(int_bbox[2], img.shape[1])

        crop_img, gt = crop_pict(img, int_bbox, picture.precomputed_gt)
        tmp = np.zeros([68, 3])
        crop_gt_2d = tmp[:, :2]
        # tmp[:, :2] = np.asarray(gt_2d)
        # _, crop_gt_2d = crop_pict(img, int_bbox, tmp)
        # crop_gt_2d = crop_gt_2d[:, :2]
        # fig, ax = plt.subplots(2, 1, figsize=(25, 25), squeeze=False)
        # ax[0, 0].imshow(img)
        # ax[0, 0].scatter(gt[:, 0], gt[:, 1], c="red", s=20)
        aug_img, aug_gt_2d = data_augment(crop_img, crop_gt_2d)
        # for id, val in enumerate(gt_2d):
        #     if np.array_equal(val, [0, 0]):
        #         aug_gt_2d[id] = [0, 0]
        # fig, ax = plt.subplots()
        # ax.imshow(aug_img)
        # ax.scatter(aug_gt[:, 0], aug_gt[:, 1], c="red", s=20)
        # ax.scatter(aug_gt_2d[:, 0], aug_gt_2d[:, 1], c="blue", s=20)
        # plt.savefig(art_folder + '/' + picture.id.split('.')[0] + '_3d&2d.png')
        plt.imsave(art_folder + '/' + picture.id.split('.')[0] + '.png', aug_img)
        # ax[1, 0].imshow(aug_img)
        # ax[1, 0].scatter(aug_gt[:, 0], aug_gt[:, 1], c="blue", s=20)
        # plt.savefig(art_folder + '/' + picture.id + '_cmp.png')
        # plt.close()
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


def make_cropped_and_3d_gt():
    with open('./dataset/folds_info.json') as file:
        folds = json.load(file)
    fold_ids = list(folds.keys())

    ds = BuddhaDataset(Config('conf_tmp.json'))
    ds.load()
    ds = ds.artifacts

    for fold_id in fold_ids:
        fold_json = {'train': {}, 'test': {}}
        for art in ds:
            if art.id in folds[fold_id]['train']:
                gt3d = art.gt
                _, gt3d = normalize_position(gt3d)
                fold_json['train'][art.id] = {'gt3d': gt3d.tolist(), 'images': {}}
                img_gt2d = process_art(fold_id, art)
                for val in img_gt2d:
                    fold_json['train'][art.id]['images'][val[0]] = val[1].tolist()
            if art.id in folds[fold_id]['test']:
                gt3d = art.gt
                _, gt3d = normalize_position(gt3d)
                fold_json['test'][art.id] = {'gt3d': gt3d.tolist(), 'images': {}}
                img_gt2d = process_art(fold_id, art)
                for val in img_gt2d:
                    fold_json['test'][art.id]['images'][val[0]] = val[1].tolist()
        with open('./dataset/' + fold_id + '/gt.json', 'w+') as file:
            json.dump(fold_json, file)


def make_intermediate_ds():
    with open('./dataset/folds_info.json') as file:
        folds = json.load(file)
    fold_ids = list(folds.keys())
    cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'
    tddfa = TDDFA_ONNX(**cfg)
    for fold_id in fold_ids:
        prediction_ds = {'train': {}, 'test': {}}
        for art in folds[fold_id]['train']:
            prediction_ds['train'][art] = {}
            imgs_path = os.listdir('./dataset/' + fold_id + '/' + art)
            for path in imgs_path:
                img_id = (path.split("/")[-1]).split('.')[0]
                img = plt.imread('./dataset/' + fold_id + '/' + art + '/' + path, )[:, :, :3]
                face_bbox = [0, 0, img.shape[0], img.shape[1]]
                param_lst, roi_box_lst = tddfa(img, [face_bbox])
                pred = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
                pred = pred[0].T
                prediction_ds['train'][art][img_id] = pred.tolist()
        for art in folds[fold_id]['test']:
            prediction_ds['test'][art] = {}
            imgs_path = os.listdir('./dataset/' + fold_id + '/' + art)
            for path in imgs_path:
                img_id = (path.split("/")[-1]).split('.')[0]
                img = plt.imread('./dataset/' + fold_id + '/' + art + '/' + path)[:, :, :3]
                face_bbox = [0, 0, img.shape[0], img.shape[1]]
                param_lst, roi_box_lst = tddfa(img, [face_bbox])
                pred = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
                pred = pred[0].T
                prediction_ds['test'][art][img_id] = pred.tolist()
        with open('./dataset/' + fold_id + '/post3DDFA.json', 'w+') as file:
            json.dump(prediction_ds, file)


if __name__ == '__main__':
    make_cropped_and_3d_gt()
    make_intermediate_ds()




