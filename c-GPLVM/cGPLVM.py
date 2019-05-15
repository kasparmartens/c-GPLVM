import torch
import torch.nn as nn
import numpy as np

from .helpers import KL_standard_normal, my_softplus

class GPLVM(nn.Module):

    def __init__(self, Y, z_init, GP_mapping, lr=1e-2, fixed_z=False, **kwargs):
        super(GPLVM, self).__init__()

        self.Y = Y
        self.output_dim = Y.size()[1]

        if fixed_z:
            self.z_mu = z_init.clone()
            self.z_logsigma = -10.0 * torch.ones_like(z_init)
        else:
            self.z_mu = nn.Parameter(z_init.clone(), requires_grad=True)
            self.z_logsigma = nn.Parameter(-1.0 * torch.ones_like(z_init), requires_grad=True)

        # for every output dimension, create a separate GP object
        self.GP_mappings = nn.ModuleList([GP_mapping(**kwargs) for j in range(self.output_dim)])

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def get_kernel_vars(self):
        return np.array([mapping.get_kernel_var().detach().numpy() for mapping in self.GP_mappings])

    def get_lengthscales(self):
        return np.array([mapping.get_ls().detach().numpy() for mapping in self.GP_mappings])

    def sample_z(self):
        eps = torch.randn_like(self.z_mu)
        z = self.z_mu + my_softplus(self.z_logsigma) * eps
        return z

    def get_z_inferred(self):
        return self.z_mu.detach().numpy()

    def optimizer_step(self):

        loss = 0.0
        # sample z
        z = self.sample_z()
        for j in range(self.output_dim):
            loss += -self.GP_mappings[j].log_prob(z, self.Y[:, j:(j + 1)])

        # KL
        loss += KL_standard_normal(self.z_mu, my_softplus(self.z_logsigma))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(self, z_star, which_kernels=None):
        N_star = z_star.size()[0]
        f_mean = torch.zeros(N_star, self.output_dim)
        f_var = torch.zeros(N_star, self.output_dim)
        for j in range(self.output_dim):
            f_mean[:, j], f_var[:, j] = self.GP_mappings[j].predict(self.z_mu, self.Y[:, j:(j + 1)], z_star,
                                                                    which_kernels)

        f_sd = torch.sqrt(1e-4 + f_var)

        return f_mean.detach(), f_sd.detach()

    def train(self, n_iter, verbose=200):

        for t in range(n_iter):

            loss = self.optimizer_step()

            if t % verbose == 0:
                print(loss)


class cGPLVM(nn.Module):

    def __init__(self, x, Y, z_init, GP_mapping, lr=1e-2, fixed_z=False, **kwargs):
        super(cGPLVM, self).__init__()

        self.Y = Y
        self.x = x
        self.output_dim = Y.size()[1]

        if fixed_z:
            self.z_mu = z_init.clone()
            self.z_logsigma = -10.0 * torch.ones_like(z_init)
        else:
            self.z_mu = nn.Parameter(z_init.clone(), requires_grad=True)
            self.z_logsigma = nn.Parameter(-1.0 * torch.ones_like(z_init), requires_grad=True)

        # for every output dimension, create a separate GP object
        self.GP_mappings = nn.ModuleList([GP_mapping(**kwargs) for j in range(self.output_dim)])

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def get_kernel_vars(self):
        return np.array([mapping.get_kernel_var().detach().numpy() for mapping in self.GP_mappings])

    def get_lengthscales(self):
        return np.array([mapping.get_ls().detach().numpy() for mapping in self.GP_mappings])

    def get_z_inferred(self):
        return self.z_mu.detach().numpy()

    def sample_z(self):
        eps = torch.randn_like(self.z_mu)
        z = self.z_mu + my_softplus(self.z_logsigma) * eps
        return z

    def optimizer_step(self):

        loss = 0.0
        # sample z
        z = self.sample_z()
        for j in range(self.output_dim):
            loss += self.GP_mappings[j].total_loss(z, self.x, self.Y[:, j:(j + 1)])

        # KL
        loss += KL_standard_normal(self.z_mu, my_softplus(self.z_logsigma))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(self, z_star, x_star):
        N_star = z_star.size()[0]
        f_mean = torch.zeros(N_star, self.output_dim)
        f_var = torch.zeros(N_star, self.output_dim)
        for j in range(self.output_dim):
            f_mean[:, j], f_var[:, j] = self.GP_mappings[j].predict(self.z_mu, self.x, self.Y[:, j:(j + 1)], z_star,
                                                                    x_star)

        f_sd = torch.sqrt(1e-6 + f_var)

        return f_mean.detach(), f_sd.detach()

    def predict_decomposition(self, z_star, x_star, which_kernels=None):
        N_star = z_star.size()[0]
        f_mean = torch.zeros(N_star, self.output_dim)
        f_var = torch.zeros(N_star, self.output_dim)
        for j in range(self.output_dim):
            f_mean[:, j], f_var[:, j] = self.GP_mappings[j].predict_decomposition(self.z_mu, self.x,
                                                                                  self.Y[:, j:(j + 1)], z_star, x_star,
                                                                                  which_kernels)

        f_sd = torch.sqrt(1e-6 + f_var)

        return f_mean.detach(), f_sd.detach()

    def train(self, n_iter, verbose=200):

        for t in range(n_iter):

            loss = self.optimizer_step()

            if t % verbose == 0:
                print("Iter {0}. Loss {1}".format(t, loss))
