import numpy as np
import pickle
import torch
from torch import nn
from torch.nn import functional as F


def convert_ds(ds):
    def same_group(i, j):
        groups = [np.arange(0, 17), np.arange(48, 68), np.arange(27, 36), np.arange(36, 42), np.arange(17, 22), np.arange(42, 48), np.arange(22, 27)]
        for group in groups:
            if i in group:
                return j in group

    graph_ds = []
    for artifact in ds:
        list_pts = []
        nb_art = len(artifact['imgs'])
        nb_nodes = 68*(nb_art + 1)
        adj_mat = np.zeros([nb_nodes, nb_nodes])
        for i in range(nb_nodes):
            for j in range(nb_nodes):
                # connect within view and homologue points in other views
                adj_mat[i, j] = 1 if i//68 == j//68 or i%68 == j%68 else 0
                # remove connections with different groups within view
                if not same_group(i%68, j%68) and i//68 == j//68:
                    adj_mat[i, j] = 0
        edges = [[], []]
        for lign_index, lign in enumerate(adj_mat):
            indexes = np.where(lign == 1)
            edges[0].append([lign_index]*len(indexes))
            edges[1].append(indexes)
        print(np.asarray(edges).shape)
        for img in artifact['imgs']:
            for pt in img['img_ldk']:
                list_pts.append(pt)
        for _ in range(68):
            list_pts.append([0, 0, 0])
        graph_ds.append([torch.as_tensor(list_pts, dtype=torch.float), adj_mat])
    return graph_ds


def get_neighbors(x, adj_mat):
    list_neighbors = []
    for lign in adj_mat:
        indexes = np.where(lign == 1)
        neighbors = x[indexes]
        list_neighbors.append(torch.as_tensor(np.expand_dims(neighbors.T, axis=0), dtype=torch.float))
    return torch.unsqueeze(x, dim=-1), list_neighbors


class GNN(nn.Module):
    def __init__(self, nb_channels=3):
        super().__init__()
        self.nb_channels = nb_channels
        self.neighbor_embeding = nn.Sequential(nn.Conv1d(in_channels=self.nb_channels, out_channels=self.nb_channels, kernel_size=(1,), stride=(1,)), nn.Tanh(),
                                               nn.Conv1d(in_channels=self.nb_channels, out_channels=self.nb_channels, kernel_size=(1,), stride=(1,)), nn.Tanh())
        self.summarize_neighbors = F.avg_pool1d
        self.self_embeding = nn.Sequential(nn.Conv1d(in_channels=self.nb_channels, out_channels=self.nb_channels, kernel_size=(1,), stride=(1,)), nn.Tanh())
        self.combine = nn.Sequential(nn.Conv1d(in_channels=self.nb_channels, out_channels=self.nb_channels, kernel_size=(2,), stride=(2,)), nn.Tanh())
        self.activation = nn.Tanh()

    def forward(self, x, list_neighbors):
        x = self.self_embeding(x)
        for index, neighbors in enumerate(list_neighbors):
            list_neighbors[index] = self.neighbor_embeding(neighbors)
            list_neighbors[index] = self.summarize_neighbors(neighbors, kernel_size=neighbors.size()[-1])
            break
        list_neighbors = torch.as_tensor(list_neighbors)
        x = self.combine(torch.cat((x, list_neighbors), dim=2))
        return self.activation(x)


class GraphNet(nn.Module):
    def __init__(self, nodes_per_view=68, nb_channels=3, depth=3):
        super().__init__()
        self.nodes_per_view = nodes_per_view
        self.nb_channels = nb_channels
        self.depth = depth
        self.GNN = [*[GNN(self.nb_channels) for _ in range(self.depth)]]

    def position_embeding(self, x):
        return x

    def forward(self, input):
        x, adj_mat = input
        x = self.position_embeding(x)
        for step in range(self.depth):
            x, list_neighbors = get_neighbors(x, adj_mat)
            x = self.GNN[step](x, list_neighbors)
        return x[-self.nodes_per_view:]


if __name__ == '__main__':
    with open('ds_full.pkl', 'rb') as f:
        ds = pickle.load(f)
    ds_train, test_val = ds
    new_ds_train = [i for i in ds_train if len(i['imgs']) != 0]
    graph_ds = convert_ds(new_ds_train[:1])
    model = GraphNet()
    out = model(graph_ds[0])
    # print(out)
