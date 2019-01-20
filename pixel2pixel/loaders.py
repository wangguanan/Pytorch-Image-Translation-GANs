'''
load train and test dataset
'''

import glob
import os

import torch.utils.data as data
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class Loaders:
    '''
    Initialize dataloaders
    '''

    def __init__(self, config):

        self.dataset_path = config.dataset_path
        self.image_size = config.image_size
        self.batch_size = config.batch_size

        self.transforms = transforms.Compose([transforms.Resize((self.image_size, self.image_size), Image.BICUBIC),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        train_set = ImageFolder(os.path.join(self.dataset_path, 'train/'), self.transforms)
        test_set = ImageFolder(os.path.join(self.dataset_path, 'test/'), self.transforms)

        self.train_loader = data.DataLoader(dataset=train_set, batch_size=self.batch_size, shuffle=True, num_workers=4, drop_last=True)
        self.test_loader = data.DataLoader(dataset=test_set, batch_size=self.batch_size, shuffle=False, num_workers=4, drop_last=False)


class ImageFolder(Dataset):
    '''
    Load images given the path
    '''

    def __init__(self, path, transform):
        self.transform = transform
        self.samples = sorted(glob.glob(os.path.join(path + '/*.*')))

    def __getitem__(self, index):

        sample = Image.open(self.samples[index])

        w, h = sample.size
        sample_target = sample.crop((0, 0, w/2, h))
        sample_source = sample.crop((w/2, 0, w, h))

        sample_source = self.transform(sample_source)
        sample_target = self.transform(sample_target)

        return sample_source, sample_target

    def __len__(self):
        return len(self.samples)
