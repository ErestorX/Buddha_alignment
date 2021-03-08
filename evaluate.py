import os
import sys
import cv2
import yaml
from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from TDDFA_ONNX import TDDFA_ONNX
from utils.functions import draw_landmarks, get_suffix
import json
import numpy as np
from buddha_loader import load_ds
import random


def get_data_from_ids(dir_path, ids):
    picture_list = os.listdir(dir_path)
    data_list = []
    for id in ids:
        data_list.append(cv2.imread(os.path.join(dir_path, picture_list[id])))
    return data_list


def load_model_ds(ds_path):
    ds = {}
    depth_prefix = 'depth_'
    img_prefix = 'vid_'
    folders = os.listdir(ds_path)
    img_ids = []
    for dir in folders:
        img_split = dir.split(img_prefix)
        if len(img_split) > 1:
            img_ids.append(img_split[1])
    for id in img_ids:
        ids = random.sample(range(0, 175), 8)
        depth_data = get_data_from_ids(os.path.join(ds_path, depth_prefix + id), ids)
        img_data = get_data_from_ids(os.path.join(ds_path, img_prefix + id), ids)
        ds[id] = {'depth': depth_data, 'img': img_data, 'ids': ids}
    return ds


if __name__ == '__main__':
    ds = load_model_ds('BlenderFiles/model_frames')
    # ds = load_ds('data') # load Buddha dataset, second arg default to True to remove singleton
    cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'
    tddfa = TDDFA_ONNX(**cfg)
    face_boxes = FaceBoxes_ONNX()
    for artifact_id in ds:
        for img, id in zip(ds[artifact_id]['img'], ds[artifact_id]['ids']):
            wfp = f'examples/results/' + artifact_id + '_' + str(id) + '_2d_sparse.jpg'

            boxes = face_boxes(img)
            n = len(boxes)
            if n == 0:
                print(f'No face detected, exit')
                sys.exit(-1)
            print(f'Detect {n} faces')

            param_lst, roi_box_lst = tddfa(img, boxes)
            ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
            draw_landmarks(img, ver_lst, dense_flag=False, wfp=wfp)
