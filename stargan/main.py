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

    # dataset
    loaders = Loaders(config)

    # main
    solver = Solver(config, loaders)
    solver.train()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Environment Configuration
    parser.add_argument('--cuda', type=str, default='4,5,6,7')
    parser.add_argument('--output_path', type=str, default='out/celeba')

    # Dataset Configuration
    parser.add_argument('--dataset_path', type=str, default='data/img_align_celeba/')
    parser.add_argument('--attr_path', type=str, default='data/list_attr_celeba.txt')
    parser.add_argument('--selected_attrs', nargs='+', type=str,
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
    parser.add_argument('--crop_size', type=int, default=178)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)

    # Model Configuration
    parser.add_argument('--class_num', type=int, default=5)
    parser.add_argument('--conv_dim', type=int, default=64)
    parser.add_argument('--layer_num', type=int, default=6)

    # Weights Configuration
    parser.add_argument('--lambda_cls', type=float, default=1)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--lambda_rec', type=float, default=10)
    parser.add_argument('--n_critics', type=int, default=5)

    # main function
    config = parser.parse_args()
    main(config)
