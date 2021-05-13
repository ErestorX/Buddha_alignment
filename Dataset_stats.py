import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from buddha_dataset import BuddhaDataset, Artifact, Image, Config, get_transform, ldk_on_im

if __name__ == "__main__":
    config = Config('conf.json')
    config.remove_singleton = False
    ds = BuddhaDataset(config)
    ds.load()
    nb_artifacts = len(ds.artifacts)
    faces_per_artifacts = np.asarray([len(art.pictures) for art in ds.artifacts])
    fig, ax = plt.subplots()
    ax.boxplot(faces_per_artifacts, vert=False)
    plt.xticks(np.arange(0, faces_per_artifacts.max() + 1, step=5))
    ax.set_title('Number of pictures per artifacts')
    plt.savefig("Number_of_pictures_per_artifacts.png")
    chart_f_per_a = [0]*(faces_per_artifacts.max() + 1)
    for val in faces_per_artifacts:
        chart_f_per_a[val] += 1
    fig, ax = plt.subplots()
    ax.plot(chart_f_per_a)
    ax.set_xlabel('Number of pictures')
    plt.xticks(np.arange(0, len(chart_f_per_a) + 1, step=5))
    ax.set_ylabel('Number of artifacts')
    ax.set_title('Artifacts per number of pictures')
    plt.savefig("Artifacts_per_number_of_pictures.png")
