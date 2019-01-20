import torch
import torch.nn as nn
from torchvision.utils import save_image
import os
import numpy as np

from models import Generator, Discriminator


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
            os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda
            self.device = torch.device('cpu')

        self._init_models()
        self._init_optimizers()


    def _init_models(self):

        self.generator = Generator(self.config.class_num, self.config.conv_dim, self.config.layer_num)
        self.discriminator = Discriminator(self.config.image_size, self.config.conv_dim, self.config.layer_num, self.config.class_num)

        self.generator = torch.nn.DataParallel(self.generator).to(self.device)
        self.discriminator = torch.nn.DataParallel(self.discriminator).to(self.device)


    def criterion_cls(self, logit, target):
        return nn.functional.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)


    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)


    def _init_optimizers(self):
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=[0.5, 0.999])
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0001, betas=[0.5, 0.999])


    def _save_model(self, current_epoch):
        torch.save(self.generator.state_dict(), os.path.join(self.save_models_path, 'G_{}.pkl'.format(current_epoch)))
        torch.save(self.discriminator.state_dict(), os.path.join(self.save_models_path, 'D_{}.pkl'.format(current_epoch)))
        print 'Note: Successfully save model as {}'.format(current_epoch)


    def _restore_model(self, resume_epoch):
        self.discriminator.load_state_dict(torch.load(os.path.join(self.save_models_path, 'D_{}.pkl'.format(resume_epoch))))
        self.generator.load_state_dict(torch.load(os.path.join(self.save_models_path, 'G_{}.pkl'.format(resume_epoch))))
        print 'Note: Successfully resume model from {}'.format(resume_epoch)


    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)


    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out


    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list


    def train(self):

        # fixed_real_images, fixed_real_labels = self.loaders.train_iter.next_one()
        # fixed_target_labels = fixed_real_labels[torch.randperm(fixed_real_labels.size(0))]
        # fixed_real_images, fixed_real_labels, fixed_target_labels = \
        #     fixed_real_images.to(self.device), fixed_real_labels.to(self.device), fixed_target_labels.to(self.device)

        x_fixed, c_org = self.loaders.train_iter.next_one()
        x_fixed = x_fixed.to(self.device)
        c_fixed_list = self.create_labels(c_org, 5, 'CelebA', ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])


        for iteration in xrange(200000):

            #########################################################################################################
            #                                                     Save Images                                       #
            #########################################################################################################
            if iteration % 100 == 0:
                # with torch.no_grad():
                #     self.generator = self.generator.eval()
                #     all_images = copy.deepcopy(fixed_real_images)
                #     for i in range(self.config.class_num):
                #         target_labels = copy.deepcopy(fixed_target_labels)
                #         target_labels[:, i] = 1
                #         fake_images = self.generator(fixed_real_images, target_labels)
                #         all_images = torch.cat([all_images, fake_images], dim=0)
                #     save_image((all_images.cpu()+1.0)/2.0,
                #                os.path.join(self.save_images_path, 'images_{}.jpg'.format(iteration)), self.config.batch_size)

                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        x_fake_list.append(self.generator(x_fixed, c_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.save_images_path, '{}-images.jpg'.format(iteration+1))
                    save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))


            #########################################################################################################
            #                                      Load Batch Data                                                  #
            #########################################################################################################
            self.generator = self.generator.train()
            real_images, real_labels = self.loaders.train_iter.next_one()
            target_labels = real_labels[torch.randperm(real_labels.size(0))]
            real_images, real_labels, target_labels = real_images.to(self.device), real_labels.to(self.device), target_labels.to(self.device)


            #########################################################################################################
            #                                                     Discriminator                                     #
            #########################################################################################################
            out_src, out_cls = self.discriminator(real_images)
            d_loss_real = -torch.mean(out_src)
            d_loss_cls = self.criterion_cls(out_cls, real_labels)

            fake_images = self.generator(real_images, target_labels)
            out_src, _ = self.discriminator(fake_images.detach())
            d_loss_fake = torch.mean(out_src)

            alpha = torch.rand(real_images.size(0), 1, 1, 1).to(self.device)
            images_hat = (alpha * real_images.data + (1 - alpha) * fake_images.data).requires_grad_(True)
            out_src, _ = self.discriminator(images_hat)
            d_loss_gp = self.gradient_penalty(out_src, images_hat)

            d_loss = d_loss_real + d_loss_fake + \
                     self.config.lambda_cls * d_loss_cls + \
                     self.config.lambda_gp * d_loss_gp

            self.d_optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()

            #########################################################################################################
            #                                                     Generator                                         #
            #########################################################################################################

            if iteration % self.config.n_critics == 0:
                fake_images = self.generator(real_images, target_labels)
                out_src, out_cls = self.discriminator(fake_images)
                g_loss_fake = -torch.mean(out_src)
                g_loss_cls = self.criterion_cls(out_cls, target_labels)

                rec_images = self.generator(fake_images, real_labels)
                g_loss_rec = torch.mean(torch.abs(real_images-rec_images))

                g_loss = g_loss_fake + self.config.lambda_rec * g_loss_rec + self.config.lambda_cls * g_loss_cls

                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()


                print 'Iter: {},  [D] gan: {}, cls: {}, gp: {};  [G]: gan: {}, rec: {},  cls: {}'.\
                    format(iteration, (d_loss_real+d_loss_fake).data, d_loss_cls.data, d_loss_gp.data,
                           g_loss_fake.data, g_loss_rec.data, g_loss_cls.data)

            if iteration % 1000 == 0:
                self._save_model(iteration)

