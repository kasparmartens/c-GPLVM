import torch
import torch.nn as nn
import numpy as np

from .helpers import KL_standard_normal, my_softplus
from .helpers_survival import calculate_KLqp_TruncatedWeibullNormal, rsample_TruncatedNormal

class survival_cGPLVM(nn.Module):

    def __init__(self, x, Y, z_init, is_censored, lower, upper, GP_mapping, fixed_z=False, lr=1e-2, **kwargs):
        """
        Initialise survival c-GPLVM (i.e. assuming some covariates c are censored)
        :param x: [N, 1] matrix of survival times (for censored data points, the corresponding entries are ignored)
        :param Y: Observed [N, P] data matrix
        :param z_init: [N, 1] matrix of initialisations for z
        :param is_censored: boolean [N, 1] indicating whether survival was observed or censored
        :param lower: lower bounds for survival (entries for non-censored data points are ignored), shape [N, 1]
        :param upper: upper bounds for survival (entries for non-censored data points are ignored), shape [N, 1]
        :param GP_mapping: which GP_mapping to use, e.g. GP_2D_AddInt
        :param fixed_z: whether z should be fixed or not
        :param lr: learning rate for Adam
        """
        super(survival_cGPLVM, self).__init__()

        N = Y.size()[0]
        self.Y = Y
        self.x_obs = x
        self.output_dim = Y.size()[1]

        if fixed_z:
            self.z_mu = z_init.clone()
            self.z_logsigma = -10.0 * torch.ones_like(z_init)
        else:
            self.z_mu = nn.Parameter(z_init.clone(), requires_grad=True)
            self.z_logsigma = nn.Parameter(-1.0 * torch.ones_like(z_init), requires_grad=True)

        Y_colmeans = Y.mean(axis=0)

        # for every output dimension, create a separate GP object
        self.GP_mappings = nn.ModuleList([GP_mapping(intercept_init=Y_colmeans[j], **kwargs) for j in range(self.output_dim)])

        # censoring related quantities
        self.is_censored = is_censored
        self.prior_shape = 2.0 * torch.ones(N, 1)
        self.prior_scale = 2.0 * torch.ones(N, 1)
        self.lower = lower
        self.upper = upper
        self.q_shape = nn.Parameter(-1.0 + torch.zeros(N, 1))
        self.q_logscale = nn.Parameter(-1.0 + torch.zeros(N, 1))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def get_kernel_vars(self):
        return np.array([mapping.get_kernel_var().detach().numpy() for mapping in self.GP_mappings])

    def get_noise_sd(self):
        return np.array([mapping.get_noise_var().sqrt().detach().numpy() for mapping in self.GP_mappings])

    def get_lengthscales(self):
        return np.array([mapping.get_ls().detach().numpy() for mapping in self.GP_mappings])

    def sample_z(self):
        eps = torch.randn_like(self.z_mu)
        z = self.z_mu + my_softplus(self.z_logsigma) * eps
        return z

    def KL_z(self):
        return KL_standard_normal(self.z_mu, my_softplus(self.z_logsigma))

    def loglik(self):

        # sample z
        z = self.sample_z()

        # for censored observations, sample x from q(surv)
        x_sample = self.sample_x()

        # for data points where "is_censored == True", set x <- x_sample, otherwise set x <- x_obs
        x = torch.zeros_like(self.x_obs)
        x[self.is_censored, :] = x_sample[self.is_censored, :]
        x[~self.is_censored, :] = self.x_obs[~self.is_censored, :]

        loss = 0.0
        for j in range(self.output_dim):
            loss += self.GP_mappings[j].total_loss(z, x, self.Y[:, j:(j + 1)])

        return loss

    def optimizer_step(self):

        loss = self.loglik() + self.KL_z() + self.KLqp_survival_helper()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict(self, z_star, x_star, add_likelihood_variance=False, to_numpy=False):
        N_star = z_star.size()[0]
        f_mean = torch.zeros(N_star, self.output_dim)
        f_var = torch.zeros(N_star, self.output_dim)
        for j in range(self.output_dim):
            x_sample = self.sample_x()
            f_mean[:, j], f_var[:, j] = self.GP_mappings[j].predict(self.z_mu, x_sample, self.Y[:, j:(j + 1)], z_star,
                                                                    x_star, add_likelihood_variance)

        f_sd = torch.sqrt(1e-6 + f_var)

        if to_numpy:
            f_mean, f_sd = f_mean.detach().numpy(), f_sd.detach().numpy()

        return f_mean, f_sd

    def predict_decomposition(self, z_star, x_star, which_kernels=None, to_numpy=False):
        N_star = z_star.size()[0]
        f_mean = torch.zeros(N_star, self.output_dim)
        f_var = torch.zeros(N_star, self.output_dim)
        for j in range(self.output_dim):
            x_sample = self.sample_x()
            f_mean[:, j], f_var[:, j] = self.GP_mappings[j].predict_decomposition(self.z_mu, x_sample,
                                                                                  self.Y[:, j:(j + 1)], z_star, x_star,
                                                                                  which_kernels)
        f_sd = torch.sqrt(1e-6 + f_var)

        if to_numpy:
            f_mean, f_sd = f_mean.detach().numpy(), f_sd.detach().numpy()

        return f_mean, f_sd

    def get_q_shape(self):
        return self.lower + (self.upper - self.lower) * torch.sigmoid(self.q_shape)

    def get_q_scale(self):
        upper_bound = 2.0
        return 1e-2 + upper_bound * torch.sigmoid(self.q_logscale)

    def KLqp_survival_helper(self):
        return calculate_KLqp_TruncatedWeibullNormal(self.prior_shape, self.prior_scale, self.get_q_shape(),
                                                     self.get_q_scale(), self.lower, self.upper, n_samples=20)

    # sample x from q(surv)  (makes sense for censored observations only)
    def sample_x(self):
        return rsample_TruncatedNormal(self.get_q_shape(), self.get_q_scale(), self.lower, self.upper)[0, :]

    def train(self, n_iter, verbose=200):

        for t in range(n_iter):

            loss = self.optimizer_step()

            if t % verbose == 0:
                print("Iter {0}. Loss {1}".format(t, loss))

