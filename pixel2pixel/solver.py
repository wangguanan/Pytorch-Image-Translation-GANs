import torch
import torch.optim as optim
from torchvision.utils import save_image
import os

from models import Generator, Discriminator, weights_init_normal


class Solver:

    def __init__(self, config, loaders):

        # Parameters
        self.config = config
        self.loaders = loaders
        self.save_images_path = os.path.join(self.config.output_path, 'images/')
        self.save_models_path = os.path.join(self.config.output_path, 'models/')

        # Set Devices
        if self.config.cuda is not '-1':
            os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Initialize
        self._init_models()
        self._init_losses()
        self._init_optimizers()

        # Resume Model
        if self.config.resume_epoch == -1:
            self.start_epoch = 0
        elif self.config.resume_epoch >=0:
            self.start_epoch = self.config.resume_epoch
            self._restore_model(self.config.resume_epoch)


    def _init_models(self):

        # Init Model
        self.generator = Generator()
        self.discriminator = Discriminator(self.config.conv_dim, self.config.layer_num)
        # Init Weights
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)
        # Move model to device (GPU or CPU)
        self.generator = torch.nn.DataParallel(self.generator).to(self.device)
        self.discriminator = torch.nn.DataParallel(self.discriminator).to(self.device)


    def _init_losses(self):
        # Init GAN loss and Reconstruction Loss
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_recon = torch.nn.L1Loss()


    def _init_optimizers(self):
        # Init Optimizer. Use Hyper-Parameters as DCGAN
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=[0.5, 0.999])
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=[0.5, 0.999])
        # Set learning-rate decay
        self.g_lr_decay = optim.lr_scheduler.StepLR(self.g_optimizer, step_size=100, gamma=0.1)
        self.d_lr_decay = optim.lr_scheduler.StepLR(self.d_optimizer, step_size=100, gamma=0.1)


    def _lr_decay_step(self, current_epoch):
        self.g_lr_decay.step(current_epoch)
        self.d_lr_decay.step(current_epoch)


    def _save_model(self, current_epoch):
        # Save generator and discriminator
        torch.save(self.generator.state_dict(), os.path.join(self.save_models_path, 'G_{}.pkl'.format(current_epoch)))
        torch.save(self.discriminator.state_dict(), os.path.join(self.save_models_path, 'D_{}.pkl'.format(current_epoch)))
        print 'Note: Successfully save model as {}'.format(current_epoch)


    def _restore_model(self, resume_epoch):
        # Resume generator and discriminator
        self.discriminator.load_state_dict(torch.load(os.path.join(self.save_models_path, 'D_{}.pkl'.format(resume_epoch))))
        self.generator.load_state_dict(torch.load(os.path.join(self.save_models_path, 'G_{}.pkl'.format(resume_epoch))))
        print 'Note: Successfully resume model from {}'.format(resume_epoch)


    def train(self):

        # Load 16 images as fixed image for displaying and debugging
        fixed_source_images, fixed_target_images = next(iter(self.loaders.train_loader))
        ones = torch.ones_like(self.discriminator(fixed_source_images, fixed_source_images))
        zeros = torch.zeros_like(self.discriminator(fixed_source_images, fixed_source_images))
        for ii in xrange(16/self.config.batch_size-1):
            fixed_source_images_, fixed_target_images_ = next(iter(self.loaders.train_loader))
            fixed_source_images = torch.cat([fixed_source_images, fixed_source_images_], dim=0)
            fixed_target_images = torch.cat([fixed_target_images, fixed_target_images_], dim=0)
        fixed_source_images, fixed_target_images = fixed_source_images.to(self.device), fixed_target_images.to(self.device)

        # Train 200 epoches
        for epoch in xrange(self.start_epoch, 200):
            # Save Images for debugging
            with torch.no_grad():
                self.generator = self.generator.eval()
                fake_images = self.generator(fixed_source_images)
                all = torch.cat([fixed_source_images, fake_images, fixed_target_images], dim=0)
                save_image((all.cpu()+1.0)/2.0,
                           os.path.join(self.save_images_path, 'images_{}.jpg'.format(epoch)), 16)

            # Train
            self.generator = self.generator.train()
            self._lr_decay_step(epoch)
            for iteration, data in enumerate(self.loaders.train_loader):
                #########################################################################################################
                #                                            load a batch data                                          #
                #########################################################################################################
                source_images, target_images = data
                source_images, target_images = source_images.to(self.device), target_images.to(self.device)

                #########################################################################################################
                #                                                     Generator                                         #
                #########################################################################################################
                fake_images = self.generator(source_images)

                gan_loss = self.criterion_GAN(self.discriminator(fake_images, source_images), ones)
                recon_loss = self.criterion_recon(fake_images, target_images)
                g_loss = gan_loss + 100 * recon_loss

                self.g_optimizer.zero_grad()
                g_loss.backward(retain_graph=True)
                self.g_optimizer.step()

                #########################################################################################################
                #                                                     Discriminator                                     #
                #########################################################################################################
                loss_real = self.criterion_GAN(self.discriminator(target_images, source_images), ones)
                loss_fake = self.criterion_GAN(self.discriminator(fake_images.detach(), source_images), zeros)
                d_loss = (loss_real + loss_fake) / 2.0

                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                print('[EPOCH:{}/{}]  [ITER:{}/{}]  [D_GAN:{}]  [G_GAN:{}]  [RECON:{}]'.
                      format(epoch, 200, iteration, len(self.loaders.train_loader), d_loss, gan_loss, recon_loss))

            # Save model
            self._save_model(epoch)
