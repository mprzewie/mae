"""
Adapted from https://github.com/dusty-nv/pytorch-classification/blob/master/nuswide.py
"""
import os
import csv
import glob

import pandas as pd
import torch
import numpy as np

from PIL import Image


class NUSWideDataset(torch.utils.data.Dataset):
    """
    Dataloader for NUS-WIDE multi-label classification dataset
    https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html

    TODO:  support custom labels and class culling
    """

    def __init__(self, root, set, transform=None, target_transform=None):
        """
        Load either the 'trainval' or 'test' set
        """
        self.root = root
        self.path_images = os.path.join(root, 'images')
        self.set = set
        self.transform = transform
        self.target_transform = target_transform

        # load the class labels
        # with open(os.path.join(root, 'labels.txt'), 'r') as file:
        #    self.classes = file.read().splitlines()

        # load the available images
        self.fn_map = {}

        for fn in glob.glob(os.path.join(root, 'images/*.jpg')):
            tmp = os.path.basename(fn).split('_')[1]
            self.fn_map[tmp] = fn

        # load class labels from CSV
        self.images = self.read_labels()

        print(f"=> NUS-WIDE classification set={set} classes={self.classes.shape} images={len(self.images)}")

    def __getitem__(self, index):
        path, target = self.images[index]

        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)

    def read_image_list(self):
        imagelist = []
        # hash2ids = {}

        if self.set == "trainval":
            path = os.path.join(self.root, "database_img.txt")
        elif self.set == "test":
            path = os.path.join(self.root, "test_img.txt")
        else:
            raise ValueError(f"invalid set '{self.set}' (should be either 'trainval' or 'test')")

        with open(path, 'r') as f:
            for i, line in enumerate(f):
                imagelist.append(os.path.join(self.root, line.strip()))
                # line = line.split('\\')[-1]
                # start = line.index('_')
                # end = line.index('.')
                # imagelist[i] = line[start + 1:end]
                # hash2ids[line[start + 1:end]] = i
                # assert False, (line, imagelist[i], hash2ids[line[start + 1:end]])


        return imagelist

    def read_labels(self):
        images = []
        num_categories = 0
        imagelist = self.read_image_list()

        if self.set == "trainval":
            file = os.path.join(self.root, "database_label_onehot.txt")
        else:
            file = os.path.join(self.root, "test_label_onehot.txt")
        # file = os.path.join(self.root, 'classification_labels', 'classification_' + self.set + '.csv')
        print(f"=> loading {file}")

        labels = pd.read_csv(file, header=None, delimiter=" ").to_numpy(dtype=float)[:, :-1] # the last column is empty

        assert not np.isnan(labels).any(), "nan in labels"

        print(labels.shape)
        assert len(labels) == len(imagelist), (len(labels), len(imagelist))

        images = list(zip(imagelist, labels))

        assert len(images) == len(imagelist), (len(images), len(imagelist))

        self.classes = labels

        # assert False, "OK"

        # with open(file, 'r') as f:
        #     reader = csv.reader(f)
        #     rownum = 0
        #     for row in reader:
        #         if header and rownum == 0:
        #             header = row
        #             self.classes = header[1:]
        #         else:
        #             if num_categories == 0:
        #                 num_categories = len(row) - 1
        #             name = int(row[0])
        #             labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)  # BCELoss requires float
        #             labels = torch.from_numpy(labels)
        #             labels[labels == -1] = 0  # TODO should these remain as -1?
        #             name2 = self.fn_map[imagelist[name]]
        #             item = (name2, labels)
        #             images.append(item)
        #         rownum += 1
        return images

    def get_class_distribution(self):
        distribution = [0] * len(self.classes)

        for _, labels in self.images:
            for n, label in enumerate(labels):
                if label > 0:
                    distribution[n] += 1

        return distribution


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='.')
    parser.add_argument('--set', type=str, default='trainval')
    parser.add_argument('--load-data', action='store_true')
    parser.add_argument('--distribution', action='store_true')

    args = parser.parse_args()
    print(args)

    # load the dataset
    dataset = NUSWideDataset(args.data, args.set)

    # verify that all images load
    if args.load_data:
        for idx, (img, target) in enumerate(dataset):
            print(
                f"loaded image {idx}  dims={img.size}  classes={[dataset.classes[n] for n in target.nonzero(as_tuple=True)[0]]}")
            # print(f"labels:  {target}")

    # get the class distributions
    if args.distribution:
        print("=> computing class distributions:")

        distribution = dataset.get_class_distribution()
        total_labels = 0

        for n, count in enumerate(distribution):
            print(f"  class {n} {dataset.classes[n]} - {count}")
            total_labels += count

        print(
            f"=> NUS-WIDE classification set={dataset.set} classes={len(dataset.classes)} images={len(dataset.images)} labels={total_labels}")
