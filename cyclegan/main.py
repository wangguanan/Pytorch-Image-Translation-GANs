import argparse
import os

from solver import Solver
from loaders import Loaders


def main(config):

    # environments
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda
    if not os.path.exists('out/'):
        os.makedirs(os.path.join(config.output_path, 'images/'))
        os.makedirs(os.path.join(config.output_path, 'models/'))

    # init dataset
    loaders = Loaders(config)

    # main
    solver = Solver(config, loaders)
    solver.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Environment Configuration
    parser.add_argument('--cuda', type=str, default='-1', help='If -1, use cpu; 2 for single GPU(2), 2,3,4 for multi GPUS(2,3,4)')
    parser.add_argument('--output_path', type=str, default='out/apple2orange/')

    # Dataset Configuration
    parser.add_argument('--dataset_path', type=str, default='data/apple2orange/')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)

    # Training Configuration
    parser.add_argument('--resume_iteration', type=int, default=-1, help='if -1, train from scratch; if >=0, resume from the model and start to train')

    # main function
    config = parser.parse_args()
    main(config)



