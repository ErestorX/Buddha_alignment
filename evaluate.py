import os
import sys
import cv2
import yaml
from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from TDDFA_ONNX import TDDFA_ONNX
from utils.functions import draw_landmarks, get_suffix
import json


def load_ds(ds_path):
    ds = {}
    for artifact_id in os.listdir(ds_path):
        ds[artifact_id] = {}

        for image_id in os.listdir(os.path.join(ds_path, artifact_id)):
            path = os.path.join(ds_path, artifact_id, image_id)
            image = {'path': path, 'data': cv2.imread(path), 'GT': [], 'preds': []}
            ds[artifact_id][image_id] = image
    return ds


if __name__ == '__main__':
    ds = load_ds('data')
    cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'
    tddfa = TDDFA_ONNX(**cfg)
    face_boxes = FaceBoxes_ONNX()
    for artifact_id in ds:
        for image_id in ds[artifact_id]:
            old_suffix = get_suffix(image_id)
            wfp = f'examples/results/{image_id.replace(old_suffix, "")}_2d_sparse.jpg'

            boxes = face_boxes(ds[artifact_id][image_id]['data'])
            n = len(boxes)
            if n == 0:
                print(f'No face detected, exit')
                sys.exit(-1)
            print(f'Detect {n} faces')

            param_lst, roi_box_lst = tddfa(ds[artifact_id][image_id]['data'], boxes)
            ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
            draw_landmarks(ds[artifact_id][image_id]['data'], ver_lst, dense_flag=False, wfp=wfp)
