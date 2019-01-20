import glob
import os

import torch.utils.data as data
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageFolder(Dataset):
    def __init__(self, path, transform):
        self.transform = transform
        self.samples = sorted(glob.glob(os.path.join(path + '/*.*')))

    def __getitem__(self, index):
        sample = self.transform(Image.open(self.samples[index]))
        return sample

    def __len__(self):
        return len(self.samples)


class IterLoader:

    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(self.loader)

    def next_one(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)


class Loaders:

    def __init__(self, config):

        self.dataset_path = config.dataset_path
        self.image_size = config.image_size
        self.batch_size = config.batch_size

        self.transforms_train = transforms.Compose([transforms.RandomResizedCrop(self.image_size),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.transforms_test = transforms.Compose([transforms.Resize(self.image_size),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        train_a_set = ImageFolder(os.path.join(self.dataset_path, 'trainA/'), self.transforms_train)
        train_b_set = ImageFolder(os.path.join(self.dataset_path, 'trainB/'), self.transforms_train)
        test_a_set = ImageFolder(os.path.join(self.dataset_path, 'testA/'), self.transforms_test)
        test_b_set = ImageFolder(os.path.join(self.dataset_path, 'testB/'), self.transforms_test)

        self.train_a_loader = data.DataLoader(dataset=train_a_set, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True)
        self.train_b_loader = data.DataLoader(dataset=train_b_set, batch_size=self.batch_size, shuffle=True, num_workers=4,  drop_last=True)
        self.test_a_loader = data.DataLoader(dataset=test_a_set, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=False)
        self.test_b_loader = data.DataLoader(dataset=test_b_set, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=False)

        self.train_a_iter = IterLoader(self.train_a_loader)
        self.train_b_iter = IterLoader(self.train_b_loader)
