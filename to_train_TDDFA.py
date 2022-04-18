# coding: utf-8

__author__ = 'cleardusk'

import os.path as osp
import time
import numpy as np
import cv2
import torch
from torchvision.transforms import Compose
import torch.backends.cudnn as cudnn

import models
from bfm import BFMModel
from utils.io import _load
from utils.functions import (
    crop_img, parse_roi_box_from_bbox, parse_roi_box_from_landmark,
)
from utils.tddfa_util import (
    load_model, _parse_param,
    ToTensorGjz, NormalizeGjz
)

make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)


def similar_transform(pts3d, roi_box, size):
    pts3d[0, :] -= 1  # for Python compatibility
    pts3d[2, :] -= 1
    pts3d[1, :] = size - pts3d[1, :]

    sx, sy, ex, ey = roi_box
    scale_x = (ex - sx) / size
    scale_y = (ey - sy) / size
    pts3d[0, :] = pts3d[0, :] * scale_x + sx
    pts3d[1, :] = pts3d[1, :] * scale_y + sy
    s = (scale_x + scale_y) / 2
    pts3d[2, :] *= s
    pts3d[2, :] -= torch.min(pts3d[2, :]).detach()
    return pts3d


class TDDFA(object):
    """TDDFA: named Three-D Dense Face Alignment (TDDFA)"""

    def __init__(self, **kvs):
        torch.set_grad_enabled(True)

        # load BFM
        self.bfm = BFMModel(
            bfm_fp=kvs.get('bfm_fp', make_abs_path('configs/bfm_noneck_v3.pkl')),
            shape_dim=kvs.get('shape_dim', 40),
            exp_dim=kvs.get('exp_dim', 10)
        )
        self.tri = self.bfm.tri
        self.bfm.u_base = torch.from_numpy(self.bfm.u_base).cuda()
        self.bfm.w_shp_base = torch.from_numpy(self.bfm.w_shp_base).cuda()
        self.bfm.w_exp_base = torch.from_numpy(self.bfm.w_exp_base).cuda()

        # config
        self.gpu_mode = kvs.get('gpu_mode', False)
        self.gpu_id = kvs.get('gpu_id', 0)
        self.size = kvs.get('size', 120)

        param_mean_std_fp = kvs.get(
            'param_mean_std_fp', make_abs_path(f'configs/param_mean_std_62d_{self.size}x{self.size}.pkl')
        )

        # load model, default output is dimension with length 62 = 12(pose) + 40(shape) +10(expression)
        model = getattr(models, kvs.get('arch'))(
            num_classes=kvs.get('num_params', 62),
            widen_factor=kvs.get('widen_factor', 1),
            size=self.size,
            mode=kvs.get('mode', 'small')
        )
        model = load_model(model, kvs.get('checkpoint_fp'))

        if self.gpu_mode:
            cudnn.benchmark = True
            model = model.cuda(device=self.gpu_id)

        self.model = model
        self.model.eval()  # eval mode, fix BN

        # data normalization
        transform_normalize = NormalizeGjz(mean=127.5, std=128)
        transform_to_tensor = ToTensorGjz()
        transform = Compose([transform_to_tensor, transform_normalize])
        self.transform = transform

        # params normalization config
        r = _load(param_mean_std_fp)
        self.param_mean = torch.from_numpy(r.get('mean')).cuda()
        self.param_std = torch.from_numpy(r.get('std')).cuda()

        # print('param_mean and param_srd', self.param_mean, self.param_std)

    def __call__(self, img_ori, obj, **kvs):
        """The main call of TDDFA, given image and box / landmark, return 3DMM params and roi_box
        :param img_ori: the input image
        :return: param list and roi_box list
        """

        crop_policy = kvs.get('crop_policy', 'box')

        if crop_policy == 'box':
            # by face box
            roi_box = parse_roi_box_from_bbox(obj)
        elif crop_policy == 'landmark':
            # by landmarks
            roi_box = parse_roi_box_from_landmark(obj)
        else:
            raise ValueError(f'Unknown crop policy {crop_policy}')

        img = crop_img(img_ori, roi_box)
        img = cv2.resize(img, dsize=(self.size, self.size), interpolation=cv2.INTER_LINEAR)
        inp = self.transform(img).unsqueeze(0)

        if self.gpu_mode:
            inp = inp.cuda(device=self.gpu_id)

        if kvs.get('timer_flag', False):
            end = time.time()
            param = self.model(inp)
            elapse = f'Inference: {(time.time() - end) * 1000:.1f}ms'
            print(elapse)
        else:
            param = self.model(inp)
        param = param.squeeze()
        return param * self.param_std + self.param_mean, roi_box, inp


    def recon_vers(self, param, roi_box):
        def reshape_fortran(x, shape):
            if len(x.shape) > 0:
                x = x.permute(*reversed(range(len(x.shape))))
            return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))
        R, offset, alpha_shp, alpha_exp = _parse_param(param)
        base = reshape_fortran(self.bfm.u_base + self.bfm.w_shp_base @ alpha_shp + self.bfm.w_exp_base @ alpha_exp, (3, -1))
        pts3d = R @ base + offset
        return similar_transform(pts3d, roi_box, self.size).T
