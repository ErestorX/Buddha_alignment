from buddha_dataset import BuddhaDataset, Config, Artifact, Image
from torchvision.transforms import Compose
from models.mobilenet_v1 import mobilenet
from matplotlib.patches import Rectangle
from torch_geometric import nn as g_nn
import matplotlib.pyplot as plt
from bfm import BFMModel
import os.path as osp
from torch import nn
import numpy as np
import os.path
import pickle
import torch
import cv2


make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)


def get_dataset(split=0.8):
    inputs_name = 'numpy_dataset.pkl'
    if inputs_name not in os.listdir('.'):
        art_ds = BuddhaDataset(Config('conf.json'))
        art_ds.load()
        art_ds = art_ds.artifacts
        inputs = []
        for art in art_ds:
            artifact = [[], [], []]
            for img in art.pictures:
                artifact[0].append(img.data)
                artifact[1].append(img.bbox)
                artifact[2].append(img.precomputed_gt)
            inputs.append(artifact)
        with open(inputs_name, 'wb') as f:
            pickle.dump(inputs, f)
    else:
        with open(inputs_name, 'rb') as f:
            inputs = pickle.load(f)
    nb_art = len(inputs)
    np.random.seed(0)
    np.random.shuffle(inputs)
    nb_train = int(nb_art * split)
    train_inputs = inputs[:nb_train]
    test_inputs = inputs[nb_train:]
    return train_inputs, test_inputs


def crop_img(img, roi_box):
    h, w = img.shape[:2]

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx
    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
    else:
        res = np.zeros((dh, dw), dtype=np.uint8)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    return res


def points_on_image(image, bbox, pts3d):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    pts3d = pts3d.detach().cpu().numpy()[:, :2]
    pts3d = pts3d.astype(np.int32)
    ax.imshow(image)
    ax.scatter(pts3d[:, 0], pts3d[:, 1], c="red", s=10)
    ax.add_patch(Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False, edgecolor='red', linewidth=2))
    plt.show()
    plt.close(fig)


def parse_roi_box_from_bbox(bbox):
    left, top, right, bottom = bbox[:4]
    old_size = (right - left + bottom - top) / 2
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0 + old_size * 0.14
    size = int(old_size * 1.58)

    roi_box = [0] * 4
    roi_box[0] = center_x - size / 2
    roi_box[1] = center_y - size / 2
    roi_box[2] = roi_box[0] + size
    roi_box[3] = roi_box[1] + size

    return roi_box


class ToTensorGjz(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NormalizeGjz(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor.sub_(self.mean).div_(self.std)
        return tensor


def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


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


def edges_for_n_images(n):
    nb_nodes = n + 1
    Sr = np.zeros([nb_nodes, nb_nodes])
    Ss = np.ones([nb_nodes, nb_nodes])
    Se = np.zeros([nb_nodes, nb_nodes])
    Er = np.zeros([nb_nodes, nb_nodes])
    Es = np.zeros([nb_nodes, nb_nodes])
    Ee = np.ones([nb_nodes, nb_nodes])
    for i in range(nb_nodes):
        Sr[i, i] = 1
        Se[i, i] = 1
        Er[i, i] = 1
        Es[i, i] = 1
    edges_Sr = np.array(np.where(Sr == 1))
    edges_Ss = np.array(np.where(Ss == 1))
    edges_Se = np.array(np.where(Se == 1))
    edges_Er = np.array(np.where(Er == 1))
    edges_Es = np.array(np.where(Es == 1))
    edges_Ee = np.array(np.where(Ee == 1))
    edges_Sr = torch.as_tensor(edges_Sr, dtype=torch.long).cuda()
    edges_Ss = torch.as_tensor(edges_Ss, dtype=torch.long).cuda()
    edges_Se = torch.as_tensor(edges_Se, dtype=torch.long).cuda()
    edges_Er = torch.as_tensor(edges_Er, dtype=torch.long).cuda()
    edges_Es = torch.as_tensor(edges_Es, dtype=torch.long).cuda()
    edges_Ee = torch.as_tensor(edges_Ee, dtype=torch.long).cuda()

    dict_edges = {('S', 'Sr', 'R'): edges_Sr, ('S', 'Ss', 'S'): edges_Ss, ('S', 'Se', 'E'): edges_Se,
                      ('E', 'Er', 'R'): edges_Er, ('E', 'Es', 'S'): edges_Es, ('E', 'Ee', 'E'): edges_Ee}
    return dict_edges


def load_model(model, checkpoint_fp):
    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        kc = k.replace('module.', '')
        if kc in model_dict.keys():
            model_dict[kc] = checkpoint[k]
        if kc in ['fc_param.bias', 'fc_param.weight']:
            model_dict[kc.replace('_param', '')] = checkpoint[k]

    model.load_state_dict(model_dict)
    return model


class HeteroFramework(nn.Module):
    def __init__(self, num_graph_steps, train_tddfa=False, train_graph=True):
        super().__init__()
        torch.set_grad_enabled(True)
        self.train_tddfa = train_tddfa
        self.train_graph = train_graph
        self.tddfa = mobilenet(num_classes=62, widen_factor=1, size=120, mode='small')
        self.tddfa = load_model(self.tddfa, 'weights/mb1_120x120.pth')
        self.tddfa.eval()
        self.transform = Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
        r = pickle.load(open(make_abs_path('configs/param_mean_std_62d_120x120.pkl'), 'rb'))
        self.param_mean = torch.from_numpy(r.get('mean')).cuda()
        self.param_std = torch.from_numpy(r.get('std')).cuda()

        self.graph = torch.nn.ModuleList()
        self.dummy_graph = False
        for _ in range(num_graph_steps):
            conv = g_nn.HeteroConv({
                ('R', 'Sr', 'S'): g_nn.SAGEConv((-1, -1), 40),
                ('S', 'Ss', 'S'): g_nn.SAGEConv((-1, -1), 40),
                ('E', 'Se', 'S'): g_nn.SAGEConv((-1, -1), 40),
                ('R', 'Er', 'E'): g_nn.SAGEConv((-1, -1), 10),
                ('S', 'Es', 'E'): g_nn.SAGEConv((-1, -1), 10),
                ('E', 'Ee', 'E'): g_nn.SAGEConv((-1, -1), 10)
            }, aggr='mean')
            self.graph.append(conv)

        bfm = BFMModel(bfm_fp=make_abs_path('configs/bfm_noneck_v3.pkl'), shape_dim=40, exp_dim=10)
        self.u_base = torch.from_numpy(bfm.u_base).cuda()
        self.w_shp_base = torch.from_numpy(bfm.w_shp_base).cuda()
        self.w_exp_base = torch.from_numpy(bfm.w_exp_base).cuda()

    def to_cuda(self):
        self.tddfa.cuda()
        self.graph.cuda()

    def convert_pred(self, base, lin_trans, bboxes):
        images_pts = None
        for trans, bbox in zip(lin_trans[1:], bboxes):
            R_ = trans.reshape(3, -1)
            R = R_[:, :3]
            offset = R_[:, -1].reshape(3, 1)
            pts3d = R @ base + offset
            pts3d = similar_transform(pts3d, bbox, 120).T
            images_pts = pts3d.unsqueeze(0) if images_pts is None else torch.cat((images_pts, pts3d.unsqueeze(0)), dim=0)
        return images_pts

    def forward(self, images, bboxes):
        vects = {'R':torch.zeros((1, 12)).cuda(), 'S':torch.zeros((1, 40)).cuda(), 'E':torch.zeros((1, 10)).cuda()}
        list_bbox = []
        for image, bbox in zip(images, bboxes):
            roi_box = parse_roi_box_from_bbox(bbox)
            list_bbox.append(roi_box)
            img = crop_img(image, roi_box)
            img = cv2.resize(img, dsize=(120, 120), interpolation=cv2.INTER_LINEAR)
            inp = self.transform(img).unsqueeze(0).cuda()
            param = self.tddfa(inp)
            param = param.squeeze() * self.param_std + self.param_mean
            vects['R'] = torch.cat((vects['R'], param[:12].unsqueeze(0)), dim=0)
            vects['S'] = torch.cat((vects['S'], param[12:52].unsqueeze(0)), dim=0)
            vects['E'] = torch.cat((vects['E'], param[52:].unsqueeze(0)), dim=0)

        edge_index_dict = edges_for_n_images(len(images))
        vects_R_save = vects['R']
        for conv in self.graph:
            vects = conv(vects, edge_index_dict)
            vects = {key: x for key, x in vects.items()}
            vects['R'] = vects_R_save
        shp, exp = vects['S'][0].reshape(-1, 1), vects['E'][0].reshape(-1, 1)
        base = reshape_fortran(self.u_base + self.w_shp_base @ shp + self.w_exp_base @ exp, (3, -1))
        return base, vects['R'], list_bbox


if __name__ == '__main__':
    train_ds, test_ds = get_dataset()
    framework = HeteroFramework(num_graph_steps=3)
    framework.to_cuda()
    for images, bboxes, gts in test_ds[:1]:
        base, lin_trans, new_bboxes = framework(images, bboxes)
        images_points = framework.convert_pred(base, lin_trans, new_bboxes)
        for img, bbox, pts in zip(images, new_bboxes, images_points):
            points_on_image(img, bbox, pts)
        for img, bbox, pts in zip(images, bboxes, gts):
            points_on_image(img, bbox, torch.from_numpy(pts).cuda())
