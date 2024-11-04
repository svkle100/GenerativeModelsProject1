import json
import os

import matplotlib.pyplot as plt


def get_dataset_path():
    if os.path.exists('/home/sven/Documents/Uni/Gen/project1/datasets'):
        return '/home/sven/Documents/Uni/Gen/project1/datasets'
    else:
        raise FileNotFoundError('Please add your dataset path to utils.py')

def visualize_batch(batch):
    bs = batch.shape[0]
    stats = json.load(open(os.path.join(get_dataset_path(), 'stats.json')))
    batch = batch * stats["std"] + stats["mean"]
    fig, axs = plt.subplots(1, bs)
    for i in range(bs):
        axs[i].axis('off')
        axs[i].imshow(batch[i][0])
    plt.show()