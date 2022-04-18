import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import plotly.express as px
from sklearn.manifold import TSNE
import tqdm
import cv2
import os
from buddha_dataset import BuddhaDataset, Config, Artifact, Image


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


class Encoder(nn.Module):
    def __init__(self, encode_dim, grey=False):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1 if grey else 3, 8, 3, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Tanh()
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(15 * 15 * 32, 128),
            nn.Tanh(),
            nn.Linear(128, encode_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):
    def __init__(self, encode_dim, grey=False):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encode_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 15 * 15 * 32),
            nn.Tanh()
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 15, 15))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.Tanh(),
            nn.ConvTranspose2d(8, 1 if grey else 3, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.tanh(x)
        return x


def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    encoder.train()
    decoder.train()
    train_loss = []
    for image_batch in dataloader:
        image_batch = image_batch.to(device)
        encoded_data = encoder(image_batch)
        decoded_data = decoder(encoded_data)
        loss = loss_fn(decoded_data, image_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())
    return np.mean(train_loss)


def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        conc_out = []
        conc_label = []
        for image_batch in dataloader:
            image_batch = image_batch.to(device)
            encoded_data = encoder(image_batch)
            decoded_data = decoder(encoded_data)
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data


def plot_ae_outputs(encoder,decoder,n=10):
    plt.figure(figsize=(16,4.5))
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img = test_dataset[i].unsqueeze(0).to(device)
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
         rec_img  = decoder(encoder(img))
      im = img.cpu().squeeze().swapaxes(0, 2).numpy()/2 + .5
      plt.imshow(im)
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      im = rec_img.cpu().squeeze().swapaxes(0, 2).numpy()/2 + .5
      plt.imshow(im)
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.show()


def show_image(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def train(encode_dim=78, num_epochs=150, batch_size=128, lr=1e-3, split=0.8, grey=False):
    inputs_name = 'inputs' + ('grey' if grey else '') + '.pt'
    if inputs_name not in os.listdir('.'):
        if grey:
            transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128), transforms.Grayscale()])
        else:
            transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
        art_ds = BuddhaDataset(Config('conf.json'))
        art_ds.load()
        art_ds = art_ds.artifacts
        inputs = None
        for art in art_ds:
            for img in art.pictures:
                image = img.data
                bbox = img.bbox
                roi_box = parse_roi_box_from_bbox(bbox)
                img = crop_img(image, roi_box)
                img = cv2.resize(img, dsize=(120, 120), interpolation=cv2.INTER_LINEAR)
                inp = transform(img).unsqueeze(0)
                # inp = inp.swapaxes(1, 3)
                if inputs is None:
                    inputs = inp
                else:
                    inputs = torch.cat((inputs, inp), 0)
        torch.save(inputs, inputs_name)
    else:
        inputs = torch.load(inputs_name)
    train_dataset, test_dataset = torch.split(inputs, [int(len(inputs) * split), int(len(inputs) * (1 - split))])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    torch.manual_seed(0)
    encoder = Encoder(encode_dim)
    decoder = Decoder(encode_dim)
    params_to_optimize = [{'params': encoder.parameters()}, {'params': decoder.parameters()}]

    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    encoder.to(device)
    decoder.to(device)

    best_loss = float('inf')
    for epoch in range(num_epochs):
        _ = train_epoch(encoder, decoder, device, train_loader, loss_fn, optim)
        val_loss = test_epoch(encoder, decoder, device, test_loader, loss_fn)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(encoder.state_dict(), 'output/train/encoder_120x120' + ('_grey' if grey else '') + '_to_' + str(encode_dim) + '.pt')
            torch.save(decoder.state_dict(), 'output/train/decoder_120x120' + ('_grey' if grey else '') + '_to_' + str(encode_dim) + '.pt')


if __name__ == '__main__':
    grey = True
    inputs_name = 'inputs'+ ('grey' if grey else '') + '.pt'
    if inputs_name not in os.listdir('.'):
        if grey:
            transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128), transforms.Grayscale()])
        else:
            transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
        art_ds = BuddhaDataset(Config('conf.json'))
        art_ds.load()
        art_ds = art_ds.artifacts
        inputs = None
        for art in art_ds:
            for img in art.pictures:
                image = img.data
                bbox = img.bbox
                roi_box = parse_roi_box_from_bbox(bbox)
                img = crop_img(image, roi_box)
                img = cv2.resize(img, dsize=(120, 120), interpolation=cv2.INTER_LINEAR)
                inp = transform(img).unsqueeze(0)
                # inp = inp.swapaxes(1, 3)
                if inputs is None:
                    inputs = inp
                else:
                    inputs = torch.cat((inputs, inp), 0)
        torch.save(inputs, inputs_name)
    else:
        inputs = torch.load(inputs_name)

    train_size = int(len(inputs) * 0.8)
    test_size = len(inputs) - train_size
    train_dataset, test_dataset = torch.split(inputs, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)
    valid_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)

    loss_fn = torch.nn.MSELoss()
    lr = 0.001
    torch.manual_seed(0)
    encoder_dim = [16, 32, 64, 78, 128, 256]
    for encode_dim in encoder_dim:
        encoder = Encoder(encode_dim, grey=grey)
        decoder = Decoder(encode_dim, grey=grey)
        params_to_optimize = [{'params': encoder.parameters()}, {'params': decoder.parameters()}]

        optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f'Selected device: {device}')

        encoder.to(device)
        decoder.to(device)

        num_epochs = 150
        if os.path.exists('output/train/encoder_120x120_to_' + str(encode_dim) + '.pt') and os.path.exists('output/train/decoder_120x120_to_' + str(encode_dim) + '.pt'):
            encoder.load_state_dict(torch.load('output/train/encoder_120x120_to_' + str(encode_dim) + '.pt'))
            decoder.load_state_dict(torch.load('output/train/decoder_120x120_to_' + str(encode_dim) + '.pt'))
            val_loss = test_epoch(encoder, decoder, device, test_loader, loss_fn)
            print('Model with encoded dimension ' + str(encode_dim) + ' has validation loss: ' + str(val_loss.numpy()))
        else:
            diz_loss = {'train_loss': [], 'val_loss': []}
            best_loss = float('inf')
            best_epoch = 0
            for epoch in range(num_epochs):
                train_loss = train_epoch(encoder, decoder, device, train_loader, loss_fn, optim)
                val_loss = test_epoch(encoder, decoder, device, test_loader, loss_fn)
                diz_loss['train_loss'].append(train_loss)
                diz_loss['val_loss'].append(val_loss)
                if epoch % (num_epochs//5) == 0:
                    print('EPOCH {}/{}: val loss {}, best epoch: {}'.format(epoch + 1, num_epochs, val_loss, best_epoch))
                    plot_ae_outputs(encoder, decoder, n=10)
                if val_loss < best_loss:
                    best_epoch = epoch
                    best_loss = val_loss
                    torch.save(encoder.state_dict(), 'output/train/encoder_120x120_to_' + str(encode_dim) + '.pt')
                    torch.save(decoder.state_dict(), 'output/train/decoder_120x120_to_' + str(encode_dim) + '.pt')

            plt.figure(figsize=(10, 8))
            plt.semilogy(diz_loss['train_loss'], label='Train')
            plt.semilogy(diz_loss['val_loss'], label='Valid')
            plt.xlabel('Epoch')
            plt.ylabel('Average Loss')
            plt.legend()
            plt.show()

        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            images = iter(test_loader).next()
            images = images.to(device)
            latent = encoder(images)
            latent = latent.cpu()
            mean = latent.mean(dim=0)
            std = (latent - mean).pow(2).mean(dim=0).sqrt()
            latent = torch.randn(128, encode_dim) * std + mean
            latent = latent.to(device)
            img_recon = decoder(latent)
            img_recon = img_recon.cpu()/2 + .5
            fig, ax = plt.subplots()
            show_image(torchvision.utils.make_grid(img_recon[:100], 10, 5))
            plt.show()
