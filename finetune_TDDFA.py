from torch.utils.data import DataLoader, TensorDataset
from generate_dataset import BuddhaDataset, Config, Artifact, Image
import torch
import cv2
import numpy as np

from TDDFA import TDDFA
import yaml
import os


def train(model, train_loader, nb_epochs, optimizer, criterion, device):
    for epoch in range(nb_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            model.model.train()
            optimizer.zero_grad()
            face_bbox = [0, 0, images.shape[0], images.shape[1]]
            param_lst, roi_box_lst = model(images, [face_bbox])
            outputs = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0].T
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, nb_epochs, i + 1, len(train_loader), loss.item()))


def art_ds_to_pic_ds(art_ds):
    inputs, targets = [], []
    for art in art_ds:
        for img in art.pictures:
            # inputs.append(np.asarray(img.cropped_data))
            inputs.append(cv2.resize(img.cropped_data, dsize=(120, 120), interpolation=cv2.INTER_LINEAR))
            targets.append(np.asarray(img.cropped_gt))
    dataset = TensorDataset(torch.from_numpy(np.asarray(inputs)), torch.from_numpy(np.asarray(targets)))
    loader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=True)
    return loader


class Loss(torch.nn.Module):
    def __call__(self, pred, label, img_size):
        distance = torch.norm(pred - label, dim=1)
        return torch.mean(distance)


if __name__ == '__main__':
    device = torch.device('cuda:0')
    cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
    cfg['gpu_mode'] = True
    cfg['gpu_id'] = 0
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'
    tddfa = TDDFA(**cfg)
    original_ds = BuddhaDataset(Config('conf.json'))
    original_ds.load()
    original_ds = original_ds.artifacts
    image_ds = art_ds_to_pic_ds(original_ds)
    criterion = Loss()
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(tddfa.model.parameters(), lr=0.001)
    # train(tddfa, image_ds, nb_epochs=15, optimizer=optimizer, criterion=criterion, device=device)


