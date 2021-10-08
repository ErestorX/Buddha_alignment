import numpy as np
import pickle
import torch
import torch.nn.functional as F


class Feedforward(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.nb_layers = len(layers)
        self.fc = torch.nn.Sequential(
            *[torch.nn.Linear(self.layers[i], self.layers[i + 1]) for i in range(self.nb_layers - 1)])
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        return self.sigmoid(x)


class ConsensusNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 68 * 3
        self.encoding_size = 1024
        self.layers_encoder = [self.input_size, 256, 512, self.encoding_size]
        self.layers_voter = [self.encoding_size, 512, 68 * 3]
        self.encoder = Feedforward(self.layers_encoder)
        self.voter = Feedforward(self.layers_voter)

    def forward(self, list_x):
        list_encoding = None
        for x in list_x:
            x = x.flatten()
            if list_encoding is None:
                list_encoding = torch.unsqueeze(self.encoder(x), 0)
            else:
                list_encoding = torch.cat((list_encoding, torch.unsqueeze(self.encoder(x), 0)), 0)
        list_encoding = torch.unsqueeze(list_encoding, 0)
        pool = F.max_pool2d(list_encoding, kernel_size=[list_encoding.shape[1], 1])
        pool = torch.squeeze(pool)
        vote = self.voter(pool)
        return vote.reshape([68, 3])


class Loss3D:
    def __call__(self, points_x, points_y):
        loss = 0
        for x, y, in zip(points_x, points_y):
            loss += torch.sqrt((y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2 + (y[2] - x[2]) ** 2)
        return loss


class Loss2D:
    def __call__(self, points_x, list_points_y, list_scales, list_rotations, list_translations):
        loss = 0
        for points_y, scale, rotation, translation in zip(list_points_y, list_scales, list_rotations,
                                                          list_translations):
            x_proj = (points_x - translation)
            tmp = torch.linalg.inv(scale * rotation)
            x_proj = torch.matmul(x_proj, tmp)
            x_proj = x_proj[:, :2]
            loss_view = 0
            count_visible = 0
            for x, y, in zip(x_proj, points_y):
                if not torch.equal(y, torch.zeros([2])):
                    count_visible += 1
                    loss_view += torch.sqrt((y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2)
            loss += loss_view / count_visible
        return loss / len(list_points_y)


def train(model, dataset, epochs):
    loss3D = Loss3D()
    loss2D = Loss2D()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    running_loss = 0.0
    for epoch in range(epochs):
        for i, artifact in enumerate(dataset):
            inputs = torch.tensor(np.asarray([img['img_ldk'] for img in artifact['imgs']]), dtype=torch.float)
            gt3D = torch.tensor(np.asarray(artifact['art_gt']), dtype=torch.float)
            gt2D = torch.tensor(np.asarray([img['img_gt'] for img in artifact['imgs']]), dtype=torch.float)
            scales = torch.tensor(np.asarray([img['img_rot'][0] for img in artifact['imgs']]), dtype=torch.float)
            rotations = torch.tensor(np.asarray([img['img_rot'][1] for img in artifact['imgs']]), dtype=torch.float)
            translations = torch.tensor(np.asarray([img['img_rot'][2] for img in artifact['imgs']]), dtype=torch.float)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            pred = model(inputs)
            loss = loss3D(pred, gt3D) + loss2D(pred, gt2D, scales, rotations, translations)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        torch.save(model, 'models/encode_and_vote/model_epoch_' + str(epoch + 1) + '.pth')


with open('ds_full.pkl', 'rb') as f:
    ds = pickle.load(f)
ds_train, test_val = ds
model = ConsensusNet()
new_ds_train = [i for i in ds_train if len(i['imgs']) != 0]
train(model, new_ds_train, 2)
