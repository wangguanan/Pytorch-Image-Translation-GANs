import torch
import torch.optim as optim
from torchvision.utils import save_image
import os
import itertools

from models import Generator, Discriminator, weights_init_normal


class Solver:

    def __init__(self, config, loaders):

        self.config = config
        self.loaders = loaders

        self.save_images_path = os.path.join(self.config.output_path, 'images/')
        self.save_models_path = os.path.join(self.config.output_path, 'models/')

        if self.config.cuda != '-1':
            os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self._init_models()
        self._init_losses()
        self._init_optimizers()

        if self.config.resume_iteration == -1:
            self.start_iteration = 0
        elif self.config.resume_iteration >=0:
            self.start_iteration = self.config.resume_iteration
            self._restore_model(self.config.resume_iteration)


    def _init_models(self):

        self.G_A2B = Generator(64, 9)
        self.D_B = Discriminator(self.config.image_size, 64, 4)
        self.G_A2B.apply(weights_init_normal)
        self.D_B.apply(weights_init_normal)
        self.G_A2B = torch.nn.DataParallel(self.G_A2B).to(self.device)
        self.D_B = torch.nn.DataParallel(self.D_B).to(self.device)

        self.G_B2A = Generator(64, 9)
        self.D_A = Discriminator(self.config.image_size, 64, 4)
        self.G_B2A.apply(weights_init_normal)
        self.D_A.apply(weights_init_normal)
        self.G_B2A = torch.nn.DataParallel(self.G_B2A).to(self.device)
        self.D_A = torch.nn.DataParallel(self.D_A).to(self.device)


    def _init_losses(self):
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()


    def _init_optimizers(self):
        self.G_optimizer = optim.Adam(itertools.chain(self.G_B2A.parameters(),self.G_A2B.parameters()), lr=0.0002, betas=[0.5, 0.999])
        self.D_A_optimizer = optim.Adam(self.D_A.parameters(), lr=0.0002, betas=[0.5, 0.999])
        self.D_B_optimizer = optim.Adam(self.D_B.parameters(), lr=0.0002, betas=[0.5, 0.999])

        self.G_lr_decay = optim.lr_scheduler.StepLR(self.G_optimizer, step_size=50000, gamma=0.1)
        self.D_A_lr_decay = optim.lr_scheduler.StepLR(self.D_A_optimizer, step_size=50000, gamma=0.1)
        self.D_B_lr_decay = optim.lr_scheduler.StepLR(self.D_B_optimizer, step_size=50000, gamma=0.1)

    def _lr_decay_step(self, current_iteration):
        self.G_lr_decay.step(current_iteration)
        self.D_A_lr_decay.step(current_iteration)
        self.D_B_lr_decay.step(current_iteration)


    def _save_model(self, current_iteration):
        torch.save(self.G_A2B.state_dict(), os.path.join(self.save_models_path, 'G_A2B_{}.pkl'.format(current_iteration)))
        torch.save(self.G_B2A.state_dict(), os.path.join(self.save_models_path, 'G_B2A_{}.pkl'.format(current_iteration)))
        torch.save(self.D_A.state_dict(), os.path.join(self.save_models_path, 'D_A_{}.pkl'.format(current_iteration)))
        torch.save(self.D_B.state_dict(), os.path.join(self.save_models_path, 'D_B_{}.pkl'.format(current_iteration)))
        print 'Note: Successfully save model as {}'.format(current_iteration)


    def _restore_model(self, resume_iteration):
        self.G_A2B.load_state_dict(torch.load(os.path.join(self.save_models_path, 'G_A2B_{}.pkl'.format(resume_iteration))))
        self.G_B2A.load_state_dict(torch.load(os.path.join(self.save_models_path, 'G_B2A_{}.pkl'.format(resume_iteration))))
        self.D_A.load_state_dict(torch.load(os.path.join(self.save_models_path, 'D_A_{}.pkl'.format(resume_iteration))))
        self.D_B.load_state_dict(torch.load(os.path.join(self.save_models_path, 'D_B_{}.pkl'.format(resume_iteration))))
        print 'Note: Successfully resume model from {}'.format(resume_iteration)


    def train(self):

        fixed_real_a = self.loaders.train_a_iter.next_one()
        fixed_real_b = self.loaders.train_b_iter.next_one()
        ones = torch.ones_like(self.D_A(fixed_real_a))
        zeros = torch.zeros_like(self.D_A(fixed_real_b))
        for ii in xrange(16/self.config.batch_size-1):
            fixed_real_a_ = self.loaders.train_a_iter.next_one()
            fixed_real_b_ = self.loaders.train_b_iter.next_one()
            fixed_real_a = torch.cat([fixed_real_a, fixed_real_a_], dim=0)
            fixed_real_b = torch.cat([fixed_real_b, fixed_real_b_], dim=0)
        fixed_real_a, fixed_real_b = fixed_real_a.to(self.device), fixed_real_b.to(self.device)


        for iteration in range(self.start_iteration, 100000):

            # save images
            if iteration % 100 == 0:
                with torch.no_grad():
                    self.G_A2B = self.G_A2B.eval()
                    self.G_B2A = self.G_B2A.eval()
                    fake_a = self.G_B2A(fixed_real_b)
                    fake_b = self.G_A2B(fixed_real_a)
                    all = torch.cat([fixed_real_a, fake_b, fixed_real_b, fake_a], dim=0)
                    save_image((all.cpu()+1.0)/2.0,
                               os.path.join(self.save_images_path, 'images_{}.jpg'.format(iteration)), 16)
            # save model
            if iteration % 1000 == 0:
                self._save_model(iteration)

            # train
            self.G_A2B = self.G_A2B.train()
            self.G_B2A = self.G_B2A.train()
            self._lr_decay_step(iteration)

            #########################################################################################################
            #                                                     Data                                              #
            #########################################################################################################
            real_a = self.loaders.train_a_iter.next_one()
            real_b = self.loaders.train_b_iter.next_one()
            real_a, real_b = real_a.to(self.device), real_b.to(self.device)

            #########################################################################################################
            #                                                     Generator                                         #
            #########################################################################################################
            fake_a = self.G_B2A(real_b)
            fake_b = self.G_A2B(real_a)

            # gan loss
            gan_loss_a = self.criterion_GAN(self.D_A(fake_a), ones)
            gan_loss_b = self.criterion_GAN(self.D_B(fake_b), ones)
            gan_loss = (gan_loss_a + gan_loss_b) / 2.0

            # cycle loss
            cycle_loss_a = self.criterion_cycle(self.G_B2A(fake_b), real_a)
            cycle_loss_b = self.criterion_cycle(self.G_A2B(fake_a), real_b)
            cycle_loss = (cycle_loss_a + cycle_loss_b) / 2.0

            # idnetity loss
            identity_loss_a = self.criterion_identity(self.G_B2A(real_a), real_a)
            identity_loss_b = self.criterion_identity(self.G_A2B(real_b), real_b)
            identity_loss = (identity_loss_a + identity_loss_b) / 2.0

            # overall loss and optimize
            g_loss = gan_loss + 10 * cycle_loss + 5 * identity_loss

            self.G_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)
            self.G_optimizer.step()

            #########################################################################################################
            #                                                     Discriminator                                     #
            #########################################################################################################
            # discriminator a
            gan_loss_a_real = self.criterion_GAN(self.D_A(real_a), ones)
            gan_loss_a_fake = self.criterion_GAN(self.D_A(fake_a.detach()), zeros)
            gan_loss_a = (gan_loss_a_real + gan_loss_a_fake) / 2.0

            self.D_A_optimizer.zero_grad()
            gan_loss_a.backward()
            self.D_A_optimizer.step()

            # discriminator b
            gan_loss_b_real = self.criterion_GAN(self.D_B(real_b), ones)
            gan_loss_b_fake = self.criterion_GAN(self.D_B(fake_b.detach()), zeros)
            gan_loss_b = (gan_loss_b_real + gan_loss_b_fake) / 2.0

            self.D_B_optimizer.zero_grad()
            gan_loss_b.backward()
            self.D_B_optimizer.step()


            print('[ITER:{}]  [G_GAN:{}]  [CYC:{}]  [IDENT:{}]  [D_GAN:{} {}]'.
                  format(iteration, gan_loss, cycle_loss, identity_loss, gan_loss_a, gan_loss_b))

