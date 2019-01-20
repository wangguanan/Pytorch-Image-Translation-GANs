import torch
import torch.nn as nn


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


class ResidualBlock(nn.Module):
    '''Residual Block with Instance Normalization'''

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
        )

    def forward(self, x):
        return self.model(x) + x


class Generator(nn.Module):
    '''Generator with Down sampling, Several ResBlocks and Up sampling.
       Down/Up Samplings are used for less computation.
    '''

    def __init__(self, class_num, conv_dim, layer_num):
        super(Generator, self).__init__()

        self.class_num = class_num

        layers = []

        # input layer
        layers.append(nn.Conv2d(in_channels=3+class_num, out_channels=conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # down sampling layers
        current_dims = conv_dim
        for i in xrange(2):
            layers.append(nn.Conv2d(current_dims, current_dims*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(current_dims*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            current_dims *= 2

        # Residual Layers
        for i in xrange(layer_num):
            layers.append(ResidualBlock(current_dims, current_dims))

        # up sampling layers
        for i in xrange(2):
            layers.append(nn.ConvTranspose2d(current_dims, current_dims//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(current_dims//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            current_dims = current_dims//2

        # output layer
        layers.append(nn.Conv2d(current_dims, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat([1, 1, x.size(2), x.size(3)])
        x = torch.cat([x, c], dim=1)
        return self.model(x)


class Discriminator(nn.Module):
    '''Discriminator with PatchGAN'''

    def __init__(self, image_size, conv_dim, layer_num, class_num):
        super(Discriminator, self).__init__()

        layers = []

        # input layer
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))
        current_dim = conv_dim

        # hidden layers
        for i in xrange(1, layer_num):
            layers.append(nn.Conv2d(current_dim, current_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            current_dim *= 2

        self.model = nn.Sequential(*layers)

        # output layer
        # compute image source (real/fake) and image class
        kernel_size = int(image_size / 2**layer_num)
        self.conv_src = nn.Conv2d(current_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_cls = nn.Conv2d(current_dim, class_num, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        x = self.model(x)
        out_src = self.conv_src(x)
        out_cls = self.conv_cls(x)
        out_cls = out_cls.view(out_cls.size(0), out_cls.size(1))
        return out_src, out_cls
