import os
import random
import itertools


import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch
from torch.autograd import Variable
import torch.nn as nn
from PIL import Image

from neural_net import load_dataset, Net, BatchNormalization

WIDTH = 100
HEIGHT = 100
PIXELS = WIDTH * HEIGHT


BLANK_SIZE = 10


def occlusion(image, x_pos, y_pos, size):
    npimg = np.transpose(image.clone().numpy(), (1, 2, 0))
    h, w, c = npimg.shape
    x_east, x_west = max(0, x_pos - size), min(w, x_pos + size)
    y_north, y_south = max(0, y_pos - size), min(h, y_pos + size)
    for channel in range(c):
        npimg[y_north:y_south, x_east:x_west, channel] = npimg[:, :, channel].mean()

    return torchvision.transforms.ToTensor()(npimg)


def positions(num_of_points):
    if num_of_points <= PIXELS // 2:
        points = []

        while len(points) < num_of_points:
            x_pos, y_pos = random.randint(0, WIDTH - 1), random.randint(0, HEIGHT - 1)
            if (x_pos, y_pos)  not in points:
                points.append((x_pos, y_pos))
    else:
        points = list(itertools.product(range(WIDTH), range(HEIGHT)))

        while len(points) > num_of_points:
            x_pos, y_pos = random.randint(0, WIDTH - 1), random.randint(0, HEIGHT - 1)
            if (x_pos, y_pos) in points:
                points.remove((x_pos, y_pos))

    return points


def heatmap_for_image(image, model, points_num, class_num):
    x = torchvision.transforms.ToTensor()(image)
    changed = torchvision.transforms.Normalize((0, 0, 0), (1, 1, 1))(x)[None]
    original_score = model(Variable(changed))[0, class_num]
    points = positions(points_num)
    heatmap = np.zeros([WIDTH, HEIGHT])

    for coor in points:
        changed = occlusion(x, coor[0], coor[1], BLANK_SIZE)
        changed = torchvision.transforms.Normalize((0, 0, 0), (1, 1, 1))(changed)[None]
        prediction = model(Variable(changed))[0, class_num]
        heatmap[coor] = original_score - prediction

    return heatmap, original_score


def heatmap_to_image(heatmap):
    return (-255 * heatmap + 255).astype(int)


def prepare_examples_per_classes(data_path, model, classes, random_points=PIXELS, save_dir='heatmaps_predictions'):

    open(os.path.join(save_dir, 'summary.txt'), 'a').close()  # clear summary
    for fruit in os.listdir(data_path):

        image = Image.open(os.path.join(data_path, fruit, os.listdir(os.path.join(data_path, fruit))[0]))
        heatmap, fruit_score = heatmap_for_image(image, model, random_points, classes.index(fruit))
        pred_image = heatmap_to_image(heatmap)

        f = plt.figure()
        f.add_subplot(1, 2, 1)
        plt.imshow(image)
        f.add_subplot(1, 2, 2)
        plt.imshow(pred_image, cmap='gray', vmin=0, vmax=255)
        plt.savefig(os.path.join(save_dir, fruit))
        plt.show()

        with open(os.path.join(save_dir, 'summary.txt'), 'a') as summary:
            summary.write(f'Base prediction for {fruit}: {fruit_score}\n')


if __name__ == '__main__':
    model = torch.load('trained_model.h5', map_location='cpu')
    classes = tuple(sorted(os.listdir('fruits-360/Training')))
    prepare_examples_per_classes('fruits-360/Training', model, classes)
