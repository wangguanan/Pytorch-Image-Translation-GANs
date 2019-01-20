'''
Define Hyper-parameters
Init Dataset and Model
Run
'''

import argparse
import os

from solver import Solver
from loaders import Loaders


def main(config):

    # Environments Parameters
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda
    if not os.path.exists(config.output_path,):
        os.makedirs(os.path.join(config.output_path, 'images/'))
        os.makedirs(os.path.join(config.output_path, 'models/'))

    # Initialize Dataset
    loaders = Loaders(config)

    # Initialize Pixel2Pixel and train
    solver = Solver(config, loaders)
    solver.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Environment Configuration
    parser.add_argument('--cuda', type=str, default='-1', help='If -1, use cpu; if >0 use single GPU; if 2,3,4 for multi GPUS(2,3,4)')
    parser.add_argument('--output_path', type=str, default='out/facades/')

    # Dataset Configuration
    parser.add_argument('--dataset_path', type=str, default='data/facades/')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)

    # Model Configuration
    parser.add_argument('--conv_dim', type=int, default=64)
    parser.add_argument('--layer_num', type=int, default=6)

    # Train Configuration
    parser.add_argument('--resume_epoch', type=int, default=-1, help='if -1, train from scratch; if >=0, resume and start to train')

    # Test Configuration
    parser.add_argument('--test_epoch', type=int, default=100)
    parser.add_argument('--test_image', type=str, default='', help='if is an image, only translate it; if a folder, translate all images in it')

    # main function
    config = parser.parse_args()
    main(config)
