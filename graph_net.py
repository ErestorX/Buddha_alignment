import torch
import pickle
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from mpl_toolkits.mplot3d import Axes3D
from torch_geometric import nn as g_nn
from torch_geometric.loader import DataLoader


class Loss3D(nn.Module):
    def __call__(self, points_x, points_y):
        loss = 0
        points_y = points_y[:68]
        for x, y, in zip(points_x, points_y):
            loss += torch.sqrt((y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2 + (y[2] - x[2]) ** 2)
        return loss


class Loss2D(nn.Module):
    def __call__(self, points_x, points_y):
        loss = 0
        points_y = points_y[68:]
        for id_view in range(len(points_y)//68):
            pt_y = points_y[id_view*68:(id_view+1)*68]
            scale, rotation, translation = pt_y[0, 2], pt_y[0, 3:12].reshape((3, 3)), pt_y[0, 12:]
            pt_y = pt_y[:, :2]
            x_proj = (points_x - translation)
            tmp = torch.linalg.inv(scale * rotation)
            x_proj = torch.matmul(x_proj, tmp)
            x_proj = x_proj[:, :2]
            loss_view = 0
            count_visible = 0
            for x, y, in zip(x_proj, pt_y):
                if y[0] >= 0 or y[1] >= 0:
                    count_visible += 1
                    loss_view += torch.sqrt((y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2)
            loss += loss_view / count_visible
        loss = loss / (len(points_y)//68)
        return loss / (len(points_y)//68)


class Loss_fn(nn.Module):
    def __init__(self, enable_2d, weight_2d=0.5):
        super().__init__()
        self.loss3D = Loss3D()
        self.loss2D = Loss2D()
        self.enable_2d = enable_2d
        self.weight_3d = 1 - weight_2d
        self.weight_2d = weight_2d

    def __call__(self, x, y):
        if self.enable_2d:
            return self.weight_3d*self.loss3D(x, y) + self.weight_2d*self.loss2D(x, y)
        else:
            return self.loss3D(x, y)


def convert_ds(ds):
    def same_group(i, j):
        groups = [np.arange(0, 17), np.arange(48, 68), np.arange(27, 36), np.arange(36, 42), np.arange(17, 22), np.arange(42, 48), np.arange(22, 27)]
        for group in groups:
            if i in group:
                return j in group

    full_ds = []
    for art_index, artifact in enumerate(ds):
        print('artifact', art_index, 'out of', len(ds))
        edges = [[], []]
        x, y = [], []
        nb_imgs = len(artifact['imgs'])
        nb_nodes = 68*(nb_imgs + 1)
        adj_mat = np.zeros([nb_nodes, nb_nodes])
        for i in range(nb_nodes):
            for j in range(nb_nodes):
                # connect within view and homologue points in other views
                adj_mat[i, j] = 1 if i//68 == j//68 or i%68 == j%68 else 0
                # remove connections with different groups within view
                if not same_group(i%68, j%68) and i//68 == j//68:
                    adj_mat[i, j] = 0
        for lign_index, lign in enumerate(adj_mat):
            indexes = np.where(lign == 1)
            edge_start = np.ones((len(indexes[0]))) * lign_index
            edges = np.asarray([np.append(edges[0], np.squeeze(edge_start)), np.append(edges[1], np.squeeze(indexes))])
        edges = edges[:, 1:]
        for ldk_id, pt in enumerate(artifact['art_gt']):
            x.append([0, 0, 0])
            y.append(np.concatenate((pt, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])))
        for img in artifact['imgs']:
            scale, rotation, translation = img['img_rot']
            transformation = [scale] + rotation.flatten().tolist() + translation.tolist()
            for ldk_id, pt in enumerate(img['img_ldk']):
                x.append(pt)
            for pt in img['img_gt']:
                y.append(np.concatenate((pt, transformation)))
        x = torch.as_tensor(x, dtype=torch.float)
        y = torch.as_tensor(y, dtype=torch.float)
        edges = torch.as_tensor(edges, dtype=torch.long)
        art_dataset = Data(x=x, y=y, edge_index=edges)
        full_ds.append(art_dataset)
    return DataLoader(full_ds)


class Pos_Embed(nn.Module):
    def __init__(self, data_dim=3, pos_embed_dim=5):
        super().__init__()
        self.data_dim = data_dim
        self.pos_embed_dim = pos_embed_dim
        self.embedding = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.pos_embed_dim, kernel_size=(1,), stride=(1,)), nn.Tanh())

    def forward(self, x):
        B, _ = x.size()
        vect = torch.arange(B, dtype=torch.float).reshape((B, 1, 1)).cuda()
        vect = torch.squeeze(self.embedding(vect))
        x = torch.cat((x, vect), dim=1)
        return x.reshape((B, self.data_dim + self.pos_embed_dim))


class GraphNet(nn.Module):
    def __init__(self, data_dim=3, pos_embed_dim=5, embed_dim=16, nodes_per_view=68):
        super().__init__()
        self.data_dim = data_dim
        self.pos_embed_dim = pos_embed_dim
        self.embed_dim = embed_dim
        self.nodes_per_view = nodes_per_view

        self.pos_embed = Pos_Embed(self.data_dim, self.pos_embed_dim)
        self.conv1 = g_nn.SAGEConv(in_channels=self.data_dim + self.pos_embed_dim, out_channels=self.embed_dim)
        self.conv2 = g_nn.SAGEConv(in_channels=self.embed_dim, out_channels=self.embed_dim)
        self.conv3 = g_nn.SAGEConv(in_channels=self.embed_dim, out_channels=self.data_dim)

    def forward(self, art_input):
        x, edge_index = art_input.x, art_input.edge_index
        x = self.pos_embed(x)
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        return x[:self.nodes_per_view]


def create_summary_image(output, y, file_name):
    output, y = output.cpu().detach(), y.cpu().detach()
    transformations = y[68:, 2:]
    y_2d, y_3d = y[68:, :2], y[:68, :3]
    nb_2d_img = y_2d.size()[0] // 68
    nb_row = 2 + nb_2d_img//3
    fig = plt.figure(figsize=(4*nb_row, 12))
    ax = fig.add_subplot(nb_row, 3, 2, projection='3d')
    X, Y, Z = y_3d.permute(1, 0)
    ax.scatter(X, Y, Z, c='b', s=5)
    X, Y, Z = output.permute(1, 0)
    ax.scatter(X, Y, Z, c='r', s=5)
    print("Saving image:\ngt3d:", y_3d, "\noutput:", output)
    for i in range(nb_2d_img):
        transformation = transformations[68*i]
        gt_2d = y_2d[68*i:68*(i+1)]
        scale, rotation, translation = transformation[0], transformation[1:10].reshape((3, 3)), transformation[10:]
        x_proj = (output - translation)
        tmp = torch.linalg.inv(rotation) / scale
        x_proj = torch.matmul(x_proj, tmp)
        x_proj = x_proj[:, :2]
        ax = fig.add_subplot(nb_row, 3, 4 + i)
        gt_2d = torch.where(gt_2d > torch.zeros(1), gt_2d, torch.zeros(1))
        X, Y = gt_2d.permute(1, 0)
        ax.scatter(X, Y, c='b', s=5)
        X, Y = x_proj.permute(1, 0)
        ax.scatter(X, Y, c='r', s=5)
        print("gt2d:", gt_2d, "\ntransformations:", scale, rotation, translation, "\nprojected:", x_proj)
    plt.savefig(file_name)


if __name__ == '__main__':
    with open('ds_0_aug.pkl', 'rb') as f:
        # pickle dump generated in dataloader.py
        ds = pickle.load(f)
    ds_train, test_val = ds
    DS_per_art = [convert_ds(ds_train), convert_ds(test_val)]
    with open('ds_precomputed_0_aug_graph.pkl', 'wb+') as f:
        pickle.dump(DS_per_art, f)

