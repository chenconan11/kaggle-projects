import torch.utils.data as data

import pandas as pd
import numpy as np

from PIL import Image

import os
import os.path

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(labels_file):
    if labels_file is None:
        return None, None, None

    labels = pd.read_csv(labels_file)
    breed_codes = labels['breed'].astype('category')
    c = breed_codes.values
    labels['label'] = c.codes

    classes = labels['label'].values
    n_classes = len(np.unique(classes))
    classes_to_name = {code: name for code, name in zip(labels['label'].values, labels['breed'].values)}
    img_to_classes = {img: code for img, code in zip(labels['id'].values, labels['label'].values)}

    return n_classes, classes_to_name, img_to_classes


def make_dataset(root_dir, img_to_classes, mode):
    samples = []
    data_dir = os.path.expanduser(root_dir)
    if os.path.exists(data_dir):
        for path, _, file_list in os.walk(data_dir):
                for file_name in sorted(file_list):
                    if has_file_allowed_extension(file_name, IMG_EXTENSIONS):
                        img_path = os.path.join(path, file_name)
                        if mode is not 'test':
                            label = img_to_classes[file_name.split('.')[0]]
                            samples.append((img_path, label))
                        else:
                            samples.append(img_path)

    return samples


class KaggleImageFolder(data.Dataset):
    def __init__(self, root_dir, labels_file=None, transform=None, mode='test', loader=Image.open, split_p=None):
        if mode not in ['train', 'valid', 'test']:
            raise Exception('''mode must in 'train', 'valid', 'test' ''')
        if labels_file is None and mode != 'test':
            raise Exception('''labels file is None!''')

        self.n_classes, self.classes_to_name, self.img_to_classes = find_classes(labels_file)
        self.samples = make_dataset(root_dir, self.img_to_classes, mode)
        self.transform = transform
        self.mode = mode
        self.loader = loader

        if split_p is not None:
            if mode == 'test':
                raise Exception('split_p must used for train or valid mode')
            if not (0 <= split_p < 1):
                raise Exception('split_p must in [0.0, 1.0)')

            self.split_p = split_p
            split_len = int(len(self.samples) * (1 - split_p))
            self.train_samples = self.samples[:split_len]
            self.valid_samples = self.samples[split_len:]
        else:
            if mode != 'test':
                raise Exception('if all data for train, split_p=0')

    def __getitem__(self, index):
        if self.mode == 'train':
            path, target = self.train_samples[index]
        elif self.mode == 'valid':
            path, target = self.valid_samples[index]
        else:
            path = self.samples[index]

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        if self.mode == 'test':
            return sample
        else:
            return sample, target

    def __len__(self):
        if self.mode == 'test':
            return len(self.samples)
        elif self.mode == 'valid':
            return len(self.valid_samples)
        else:
            return len(self.train_samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        return str


if __name__ == '__main__':

    root = 'data'
    train_dir = root + '/train'
    test_dir = root + '/test'
    labels_csv = 'labels.csv'
    # train_data = KaggleImageFolder(train_dir, labels_file, mode='train', split_p=0.2)
    # print(len(train_data))
    # valid_data = KaggleImageFolder(train_dir, labels_file, mode='valid', split_p=0.2)
    # print(len(valid_data))
    # test_data = KaggleImageFolder(test_dir, mode='test')
    # print(len(test_data))

    train_data = KaggleImageFolder(train_dir, labels_csv, mode='train')
    print(len(train_data))
    valid_data = KaggleImageFolder(train_dir, labels_csv, mode='test', split_p=0.2)
    print(len(valid_data))
    test_data = KaggleImageFolder(test_dir, labels_csv, mode='test')
    print(len(test_data))
