import os
import cv2
import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from buddha_loader import load_ds
from TDDFA_ONNX import TDDFA_ONNX
from tensorflow.keras import layers
from mpl_toolkits.mplot3d import Axes3D


def normalize(points):
    centred = points - np.mean(points, axis=0)
    return centred / centred.max()


def apply_rand_T(vect):
    alpha, beta, gamma = .5 * np.pi * np.random.random(3)
    Rx = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    Rz = np.array([[1, 0, 0], [0, np.cos(gamma), -np.sin(gamma)], [0, np.sin(gamma), np.cos(gamma)]])
    rot = [Rx, Ry, Rz]
    trans = rot[np.random.choice(2, 1)[0]]
    tmp = vect @ trans
    tmp = np.reshape(tmp.T, [3, 68, 1])
    return tmp


class CNN:
    def __init__(self, ds):
        self.RATIO = .75
        cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'
        self.tddfa = TDDFA_ONNX(**cfg)
        data, labels = [], []
        for art_key in ds.keys():
            vect, label = self.preprocess(ds[art_key], ds[art_key]['machine_gt'] + ds[art_key]['human_gt'])
            data.append(vect)
            labels.append(apply_rand_T(label))
        data = np.asarray(data)
        labels = np.asarray(labels)
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        self.train_ds = dataset.take(int(self.RATIO * labels.shape[0])).batch(8)
        self.test_ds = dataset.skip(int(self.RATIO * labels.shape[0])).batch(8)

        inputs = keras.Input(shape=(3, 68, 64), name="digits")
        x = layers.Conv2D(32, 3, padding="same", activation="sigmoid", name="conv_1")(inputs)
        x = layers.Conv2D(16, 3, padding="same", activation="sigmoid", name="conv_2")(x)
        x = layers.Conv2D(8, 3, padding="same", activation="sigmoid", name="conv_3")(x)
        outputs = layers.Conv2D(1, 3, padding="same", activation="sigmoid", name="predictions")(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=keras.optimizers.RMSprop(), loss=keras.losses.MeanSquaredError(),
                      metrics=[keras.metrics.MeanSquaredError()], )
        log_dir = "logs/normedIn_randRotOut"
        self.file_writer_cm = tf.summary.create_file_writer(log_dir)
        self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
        self.plot_callback = keras.callbacks.LambdaCallback(on_epoch_end=self.log_3dplot)

    def preprocess(self, artifact, label):
        vect = []
        for id in artifact['pictures'].keys():
            image = artifact['pictures'][id]['cropped_data']
            face_bbox = [0, 0, image.shape[0], image.shape[1]]
            param_lst, roi_box_lst = self.tddfa(image, [face_bbox])
            pred_3d = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
            pred_3d = pred_3d[0].T
            pred_3d = normalize(pred_3d)
            orientation = pred_3d[-1] - pred_3d[0] / np.linalg.norm(pred_3d[-1] - pred_3d[0])
            vect.append(pred_3d)
        vect = np.asarray(vect)
        pad = 64 - len(vect)
        vect = np.pad(vect, ((0,pad), (0,0), (0,0)), 'constant', constant_values=0).T
        return vect, normalize(label)

    def log_3dplot(self, epoch, logs):
        nb_output = 4
        preds = self.model.predict(self.test_ds)
        gts = self.test_ds.unbatch().enumerate()
        for id, (pred, gt) in enumerate(zip(preds, gts)):
            ldk_gt = gt[1][1]
            if id >= nb_output:
                break
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(pred[0], pred[1], pred[2], c='b')
            ax.scatter(ldk_gt[0], ldk_gt[1], ldk_gt[2], c='r')
            plt.savefig("examples/tmp/" + str(id) + ".jpeg")
        fig_stack = np.asarray([cv2.imread(os.path.join("examples/tmp/", file), cv2.IMREAD_UNCHANGED) for file in os.listdir("examples/tmp/")])
        np.where(fig_stack.any(-1,keepdims=True),fig_stack,255)
        with self.file_writer_cm.as_default():
            tf.summary.image("test masks examples", fig_stack, max_outputs=nb_output, step=epoch)

    def train(self, nb_epochs):
        with tf.device('/CPU:0'):
            history = self.model.fit(self.train_ds, epochs=nb_epochs, callbacks=[self.plot_callback, self.tensorboard_callback])
            self.model.save("examples/train_saves/CNN_consensus")


if __name__ == '__main__':
    ds = load_ds('data')
    cnn = CNN(ds)
    # cnn.train(500)
