import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

from collections import namedtuple
import csv
import json
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset

def read_csv(csv_file_path):
    filenames = []
    with open(csv_file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                filenames.append(row[1])

    return filenames

class EmbeddingsDataset(Dataset):
    def __init__(self, data_dir, fold_name, vocab_file='labelvocabulary.csv'):
        vocab_path = os.path.join(data_dir, vocab_file)
        
        if os.path.exists(vocab_path):
            self.vocab_list = read_csv(vocab_path)
        else:
            raise Exception("Data folder must contain a valid vocab index csv file")

        fold_label_file = fold_name + '.json'
        label_path = os.path.join(data_dir, fold_label_file)
    
        with open(label_path, mode='r') as file:
            data = json.load(file)
            
        self.samples = list(data.keys())
        
        self.fold_name = fold_name
        self.data_dir = data_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx] + '.embedding.npy'
        embed_path = os.path.join(self.data_dir, self.fold_name, sample)
        embeddings = np.load(embed_path)
        
        embeddings = torch.from_numpy(embeddings)
        label = torch.tensor(int(sample[:3]), dtype=torch.long)
        return [embeddings, label]


class DatasetBuilder(object):
    # tuple for dataset config
    DC = namedtuple('DatasetConfig', ['mean', 'std', 'input_size', 'num_classes'])
    
    DATASET_CONFIG = {
        'svhn' :   DC([0.43768210, 0.44376970, 0.47280442], [0.19803012, 0.20101562, 0.19703614], 32, 10),
        'cifar10': DC([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784], 32, 10),
    } 

    def __init__(self, name:str, root_path:str):
        """
        Args
        - name: name of dataset
        - root_path: root path to datasets
        """
        if name not in self.DATASET_CONFIG.keys():
            raise ValueError('name of dataset is invalid')
        self.name = name
        self.root_path = os.path.join(root_path, self.name)

    def __call__(self, train:bool, normalize:bool):
        input_size = self.DATASET_CONFIG[self.name].input_size
        transform = self._get_trainsform(self.name, input_size, train, normalize)
        if self.name == 'svhn':
            dataset = torchvision.datasets.SVHN(root=self.root_path, split='train' if self.train else 'test', transform=transform, download=True)
        elif self.name == 'cifar10':
            dataset = torchvision.datasets.CIFAR10(root=self.root_path, train=train, transform=transform, download=True)
        else: 
            raise NotImplementedError 

        return dataset

    def _get_trainsform(self, name:str, input_size:int, train:bool, normalize:bool):
        transform = []

        # arugmentation
        if train:
            transform.extend([
                torchvision.transforms.RandomHorizontalFlip(),
            ])

        else:
            pass

        # to tensor
        transform.extend([torchvision.transforms.ToTensor(),])

        # normalize
        if normalize:
            transform.extend([
                torchvision.transforms.Normalize(mean=self.DATASET_CONFIG[name].mean, std=self.DATASET_CONFIG[name].std),
            ])

        return torchvision.transforms.Compose(transform)
    
    @property
    def input_size(self):
        return self.DATASET_CONFIG[self.name].input_size

    @property
    def num_classes(self):
        return self.DATASET_CONFIG[self.name].num_classes