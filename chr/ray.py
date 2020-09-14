import csv
import errno
import os
import os.path

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

object_categories = ['Gun', 'Knife', 'Wrench', 'Pliers', 'Scissors']


def read_image_label(file):
    print('[dataset] read ' + file)
    data = dict()
    with open(file, 'r') as f:
        for line in f:
            tmp = line.split(' ')
            name = tmp[0]
            label = int(tmp[-1])
            data[name] = label
            # data.append([name, label])
            # print('%s  %d' % (name, label))
    return data


def write_object_labels_csv(file, labeled_data):
    # write a csv file
    print('[dataset] write file %s' % file)
    with open(file, 'w') as csvfile:
        fieldnames = ['name']
        fieldnames.extend(object_categories)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for (name, labels) in labeled_data.items():
            example = {'name': name}
            for i in range(5):
                example[fieldnames[i + 1]] = int(labels[i])
            writer.writerow(example)

    csvfile.close()


def read_object_labels_csv(file, header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                labels = torch.from_numpy(labels)
                item = (name, labels)
                images.append(item)
            rownum += 1
    return images


class XrayClassification(data.Dataset):
    def __init__(self, root, set, subset, transform=None, target_transform=None):
        """
        Dataset loader
        Parameters
        ----------
        root: Root directory of the dataset
        set: "train" or "test" set
        subset: One of the subsets of the dataset e.g., "10"/"100"/"1000"
        transform: Apply transformation or not? (is it boolean?)
        target_transform: Apply transformation to the target values (is it boolean?)
        """
        self.root = root
        self.path_devkit = os.path.join(root)
        self.path_images = os.path.join(root, 'JPEGImage')
        self.set = set
        self.subset = subset
        self.transform = transform
        self.target_transform = target_transform

        # define filename of csv file
        csv_file_dir = os.path.join(self.root, 'ImageSet', self.subset)
        file_csv = os.path.join(csv_file_dir, set + '.csv')

        # create the csv file if necessary
        # (hassanmohsin) Following block is broken, so commenting out
        # if not os.path.exists(file_csv):
        #     if not os.path.exists(path_csv):  # create dir if necessary
        #         os.makedirs(path_csv)
        #     # generate csv file
        #     labeled_data = read_object_labels_csv(self.root, self.set)
        #     # write csv file
        #     write_object_labels_csv(file_csv, labeled_data)
        if not os.path.isfile(file_csv):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_csv)

        self.classes = object_categories
        self.images = read_object_labels_csv(file_csv)

        # Remove missing images
        with open(os.path.join(self.root, 'ImageSet', "missing_files.txt")) as f:
            missing_images = f.read().splitlines()
        # remove extensions
        missing_images = [im.split('.')[0] for im in missing_images]
        self.images = [im for im in self.images if im[0] not in missing_images]

        print('[dataset] X-ray classification set=%s number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(os.path.join(self.path_images, path + '.jpg')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img, path), target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)
