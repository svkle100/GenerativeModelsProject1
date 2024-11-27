import json
import os

import matplotlib.pyplot as plt
import torch
from IPython.core.pylabtools import figsize


def get_dataset_path():
    if os.path.exists('/home/sven/Documents/Uni/Gen/project1/datasets'):
        return '/home/sven/Documents/Uni/Gen/project1/datasets'
    else:
        raise FileNotFoundError('Please add your dataset path to utils.py')

def visualize_batch(batch):
    bs = batch.shape[0]
    stats = json.load(open(os.path.join(get_dataset_path(), 'stats.json')))
    batch = batch * stats["std"] + stats["mean"]
    batch = batch.clone().detach().cpu().numpy()
    fig, axs = plt.subplots(1, bs, figsize = [bs * 2, bs])
    for i in range(bs):
        axs[i].axis('off')
        axs[i].imshow(batch[i][0])
    plt.show()

def save_model(model, options):
    name = str(hash(model.parameters()))
    dir = os.path.join(get_dataset_path(), "../models", name)
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(os.path.join(dir, "options.json"), "w") as outfile:
        json.dump(options, outfile)
    torch.save(model.state_dict(), os.path.join(dir, "model.pt"))
