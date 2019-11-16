import torch
import torch.nn as nn
import numpy as np

from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal

from .helpers import my_softplus, grid_helper
from .kernels import RBF, meanzeroRBF, addint_2D_kernel_decomposition, addint_kernel_diag


# class for GP-regression with inputs (z, x) and 1D output y
class GP_2D_AddInt(nn.Module):

    def __init__(self, z_inducing, x_inducing, mean_zero=True, covariate_dim=1, kernel_var_init=None, ls_init=None):

        super(GP_2D_AddInt, self).__init__()

        self.mean_zero = mean_zero
        self.a = -2.0
        self.b = 2.0
        self.jitter = 1e-4

        self.covariate_dim = covariate_dim
        if covariate_dim > 1:
            raise ValueError("This implementation only supports univariate covariate")

        # kernel hyperparameters
        n_kernel_vars = 1 + 2 * covariate_dim
        n_lengthscales = 1 + 3 * covariate_dim

        # lengthscales
        if ls_init is None:
            self.ls = nn.Parameter(3.0 * torch.ones(n_lengthscales), requires_grad=True)
        else:
            self.ls = nn.Parameter(ls_init.clone(), requires_grad=True)

        # kernel variances
        if kernel_var_init is None:
            var_init = torch.cat([torch.ones(1 + covariate_dim), -1.0 + torch.zeros(covariate_dim)])
            self.var = nn.Parameter(var_init, requires_grad=True)
        else:
            self.var = nn.Parameter(kernel_var_init.clone(), requires_grad=True)

        self.noise_var = nn.Parameter(-1.0 * torch.ones(1), requires_grad=True)
        self.intercept = nn.Parameter(torch.zeros(1), requires_grad=True)

        # inducing points
        self.z_u, self.x_u = grid_helper(z_inducing, x_inducing)
        self.M = self.z_u.size()[0]

    def get_kernel_var(self):
        return 1e-4 + my_softplus(self.var)

    def get_ls(self):
        return 1e-4 + my_softplus(self.ls)

    def get_noise_var(self):
        return 1e-4 + my_softplus(self.noise_var)

    def kernel_decomposition_single_covariate(self, z, z2, x, x2, jitter=None):
        K1, K2, K3 = addint_2D_kernel_decomposition(z, z2, x, x2, self.get_ls(), self.get_kernel_var(), a=self.a,
                                                    b=self.b, mean_zero=self.mean_zero, jitter=jitter)
        return K1, K2, K3

    def kernel_decomposition(self, z, z2, x, x2, jitter=None):
        if self.covariate_dim == 1:
            return self.kernel_decomposition_single_covariate(z, z2, x, x2, jitter)
        else:
            return self.kernel_decomposition_multiple_covariates(z, z2, x, x2, jitter)

    def get_K_without_noise(self, z, z2, x, x2, jitter=None, which_kernels=None):
        K1, K2, K3 = self.kernel_decomposition(z, z2, x, x2, jitter)
        if which_kernels is None:
            return K1 + K2 + K3
        else:
            return which_kernels[0] * K1 + which_kernels[1] * K2 + which_kernels[2] * K3

    def get_K_diag(self, z, x):
        K1diag, K2diag, K3diag = addint_kernel_diag(z, x, self.get_ls(), self.get_kernel_var(), a=self.a, b=self.b,
                                                    mean_zero=self.mean_zero)
        return K1diag + K2diag + K3diag

    def log_prob(self, z, x, y):
        subset = ~torch.isnan(y).reshape(-1)
        if subset.sum() > 0:
            y = y[subset, :]
            z = z[subset, :]
            x = x[subset, :]
        # inducing points log_prob
        N = y.size()[0]
        y = (y - self.intercept)
        sigma2 = self.get_noise_var()
        sigma = torch.sqrt(sigma2)
        K_uu = self.get_K_without_noise(self.z_u, self.z_u, self.x_u, self.x_u, jitter=self.jitter)
        K_uf = self.get_K_without_noise(self.z_u, z, self.x_u, x)
        L = torch.cholesky(K_uu)
        A = torch.triangular_solve(K_uf, L, upper=False)[0]
        AAT = torch.mm(A, A.t())
        B = AAT + torch.eye(self.M)
        LB = torch.cholesky(B)
        Aerr = torch.mm(A, y)
        c = torch.triangular_solve(Aerr, LB, upper=False)[0] / sigma

        Kdiag = self.get_K_diag(z, x)

        bound = -0.5 * N * np.log(2 * np.pi) - torch.sum(torch.log(torch.diag(LB))) - 0.5 * N * torch.log(
            sigma2) - 0.5 * torch.sum(torch.pow(y, 2)) / sigma2
        bound += 0.5 * torch.sum(torch.pow(c, 2)) - 0.5 * torch.sum(Kdiag) / sigma2 + 0.5 * torch.sum(torch.diag(AAT))
        return bound

    def predict(self, z, x, y, z_star, x_star, add_likelihood_variance=False):
        subset = ~torch.isnan(y).reshape(-1)
        if subset.sum() > 0:
            y = y[subset, :]
            z = z[subset, :]
            x = x[subset, :]
        Nstar = z_star.size()[0]
        y = (y - self.intercept)
        sigma2 = self.get_noise_var()

        sigma = torch.sqrt(sigma2)

        K_uu = self.get_K_without_noise(self.z_u, self.z_u, self.x_u, self.x_u, jitter=self.jitter)
        K_uf = self.get_K_without_noise(self.z_u, z, self.x_u, x)
        K_us = self.get_K_without_noise(self.z_u, z_star, self.x_u, x_star)

        L = torch.cholesky(K_uu)
        A = torch.triangular_solve(K_uf, L, upper=False)[0] / sigma
        AAT = torch.mm(A, A.t())
        B = AAT + torch.eye(self.M)
        LB = torch.cholesky(B)
        Aerr = torch.mm(A, y)
        c = torch.triangular_solve(Aerr, LB, upper=False)[0] / sigma

        tmp1 = torch.triangular_solve(K_us, L, upper=False)[0]
        tmp2 = torch.triangular_solve(tmp1, LB, upper=False)[0]
        mean = self.intercept + torch.mm(tmp2.t(), c)

        Kdiag = self.get_K_diag(z_star, x_star)
        var = Kdiag + torch.pow(tmp2, 2).sum(dim=0) - torch.pow(tmp1, 2).sum(dim=0)

        if add_likelihood_variance:
            var += self.get_noise_var()

        return mean.reshape(-1), var

    def predict_decomposition(self, z, x, y, z_star, x_star, which_kernels):
        subset = ~torch.isnan(y).reshape(-1)
        if subset.sum() > 0:
            y = y[subset, :]
            z = z[subset, :]
            x = x[subset, :]
        Nstar = z_star.size()[0]
        y = (y - self.intercept)
        sigma2 = self.get_noise_var()

        K_all = self.get_K_without_noise(z, z, x, x, jitter=self.jitter) + sigma2 * torch.eye(y.size()[0])
        L_all = torch.cholesky(K_all)
        K_all_inv = torch.cholesky_inverse(L_all)

        K_sf = self.get_K_without_noise(z_star, z, x_star, x, which_kernels=which_kernels)
        K_ss = self.get_K_without_noise(z_star, z_star, x_star, x_star, which_kernels=which_kernels)

        tmp = torch.mm(K_sf, K_all_inv)
        mean = torch.mm(tmp, y)
        var = torch.diag(K_ss - torch.mm(tmp, K_sf.t()))

        return mean.reshape(-1), var

    def prior_loss(self):
        p_ls = Gamma(50.0, 10.0).log_prob(self.get_ls()).sum()
        p_var = Gamma(1.0, 1.0).log_prob(self.get_kernel_var()).sum()
        return -1.0 * (p_ls + p_var)

    def total_loss(self, z, x, y):
        return -self.log_prob(z, x, y) + self.prior_loss()


# GP with 2D ARD kernel on (z, x)
class GP_2D_INT(nn.Module):

    def __init__(self, z_inducing=None, x_inducing=None):

        super(GP_2D_INT, self).__init__()

        self.jitter = 1e-3

        # kernel hyperparameters
        self.ls = nn.Parameter(torch.ones(2), requires_grad=True)
        self.var = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.noise_var = nn.Parameter(-1.0 * torch.ones(1), requires_grad=True)
        self.intercept = nn.Parameter(torch.zeros(1), requires_grad=True)

        # create a grid of inducing points (assuming one-dimensional z and x)
        self.z_u, self.x_u = grid_helper(z_inducing, x_inducing)
        self.M = self.z_u.size()[0]

    def get_kernel_var(self):
        return 1e-4 + my_softplus(self.var)

    def get_ls(self):
        return 1e-4 + my_softplus(self.ls)

    def get_noise_var(self):
        return 1e-4 + my_softplus(self.noise_var)

    def get_K_without_noise(self, z, z2, x, x2, jitter=None):
        zx = torch.cat([z, x], dim=1)
        zx2 = torch.cat([z2, x2], dim=1)
        K = RBF(zx, zx2, self.get_ls(), self.get_kernel_var(), jitter)
        return K

    def log_prob(self, z, x, y):
        # inducing points log_prob
        subset = ~torch.isnan(y).reshape(-1)
        if subset.sum() > 0:
            y = y[subset, :]
            z = z[subset, :]
            x = x[subset, :]
        N = y.size()[0]
        y = (y - self.intercept)
        sigma2 = self.get_noise_var()
        sigma = torch.sqrt(sigma2)
        K_uu = self.get_K_without_noise(self.z_u, self.z_u, self.x_u, self.x_u, jitter=self.jitter)
        K_uf = self.get_K_without_noise(self.z_u, z, self.x_u, x)
        L = torch.cholesky(K_uu)
        A = torch.triangular_solve(K_uf, L, upper=False)[0] / sigma
        AAT = torch.mm(A, A.t())
        B = AAT + torch.eye(self.M)
        LB = torch.cholesky(B)
        Aerr = torch.mm(A, y)
        c = torch.triangular_solve(Aerr, LB, upper=False)[0] / sigma

        Kdiag = self.get_kernel_var().repeat(N)

        bound = -0.5 * N * np.log(2 * np.pi) - torch.sum(torch.log(torch.diag(LB))) - 0.5 * N * torch.log(
            sigma2) - 0.5 * torch.sum(torch.pow(y, 2)) / sigma2
        bound += 0.5 * torch.sum(torch.pow(c, 2)) - 0.5 * torch.sum(Kdiag) / sigma2 + 0.5 * torch.sum(torch.diag(AAT))
        return bound

    def predict(self, z, x, y, z_star, x_star, add_likelihood_variance=False):
        subset = ~torch.isnan(y).reshape(-1)
        if subset.sum() > 0:
            y = y[subset, :]
            z = z[subset, :]
            x = x[subset, :]
        Nstar = z_star.size()[0]
        y = (y - self.intercept)
        sigma2 = self.get_noise_var()
        sigma = torch.sqrt(sigma2)

        K_uu = self.get_K_without_noise(self.z_u, self.z_u, self.x_u, self.x_u, jitter=self.jitter)
        K_uf = self.get_K_without_noise(self.z_u, z, self.x_u, x)
        K_us = self.get_K_without_noise(self.z_u, z_star, self.x_u, x_star)

        L = torch.cholesky(K_uu)
        A = torch.triangular_solve(K_uf, L, upper=False)[0] / sigma
        AAT = torch.mm(A, A.t())
        B = AAT + torch.eye(self.M)
        LB = torch.cholesky(B)
        Aerr = torch.mm(A, y)
        c = torch.triangular_solve(Aerr, LB, upper=False)[0] / sigma

        tmp1 = torch.triangular_solve(K_us, L, upper=False)[0]
        tmp2 = torch.triangular_solve(tmp1, LB, upper=False)[0]
        mean = self.intercept + torch.mm(tmp2.t(), c)

        K_ss = self.get_K_without_noise(z_star, z_star, x_star, x_star)
        Kdiag = K_ss.diag()
        var = Kdiag + torch.pow(tmp2, 2).sum(dim=0) - torch.pow(tmp1, 2).sum(dim=0)

        if add_likelihood_variance:
            var += self.get_noise_var()

        return mean.reshape(-1), var

    def log_prob_fullrank(self, z, x, y):
        N = z.size()[0]
        y = (y - self.intercept).reshape(-1)
        K = self.get_K_without_noise(z, z, x, x, jitter=self.jitter, which_kernels=None)
        K_noise = self.get_noise_var() * torch.eye(N)
        return MultivariateNormal(torch.zeros_like(y), K + K_noise).log_prob(y)

    def prior_loss(self):
        prior_var = -Gamma(1.0, 1.0).log_prob(self.get_kernel_var()).sum()
        prior_ls = -Gamma(10.0, 1.0).log_prob(self.get_ls()).sum()
        return prior_var + prior_ls

    def total_loss(self, z, x, y):
        return -self.log_prob(z, x, y) + self.prior_loss()


# additive GP
class GP_2D_ADD(nn.Module):

    def __init__(self, z_inducing, x_inducing):

        super().__init__()

        self.jitter = 1e-4

        # kernel hyperparameters
        self.ls = nn.Parameter(torch.ones(2), requires_grad=True)
        self.var = nn.Parameter(torch.zeros(2), requires_grad=True)
        self.noise_var = nn.Parameter(-1.0 * torch.ones(1), requires_grad=True)
        self.intercept = nn.Parameter(torch.zeros(1), requires_grad=True)

        # inducing points
        grid = torch.linspace(-3, 3, steps=10).reshape(-1, 1)
        self.z_u, self.x_u = grid_helper(z_inducing, x_inducing)
        self.M = self.z_u.size()[0]

    def get_kernel_var(self):
        return 1e-4 + my_softplus(self.var)

    def get_ls(self):
        return 1e-4 + my_softplus(self.ls)

    def get_noise_var(self):
        return 1e-4 + my_softplus(self.noise_var)

    def kernel_decomposition(self, z, z2, x, x2, jitter=None):
        ls = self.get_ls()
        var = self.get_kernel_var()
        K1 = RBF(z, z2, ls[0], var[0], jitter)
        K2 = RBF(x, x2, ls[1], var[1], jitter)
        return K1, K2

    def get_K_without_noise(self, z, z2, x, x2, jitter=None, which_kernels=None):
        K1, K2 = self.kernel_decomposition(z, z2, x, x2, jitter)
        if which_kernels is None:
            return K1 + K2
        else:
            return which_kernels[0] * K1 + which_kernels[1] * K2

    def log_prob(self, z, x, y):
        subset = ~torch.isnan(y).reshape(-1)
        if subset.sum() > 0:
            y = y[subset, :]
            z = z[subset, :]
            x = x[subset, :]
        # inducing points log_prob

        N = y.size()[0]
        y = (y - self.intercept)
        sigma2 = self.get_noise_var()
        sigma = torch.sqrt(sigma2)
        K_uu = self.get_K_without_noise(self.z_u, self.z_u, self.x_u, self.x_u, jitter=self.jitter)
        K_uf = self.get_K_without_noise(self.z_u, z, self.x_u, x)
        L = torch.cholesky(K_uu)
        A = torch.triangular_solve(K_uf, L, upper=False)[0] / sigma
        AAT = torch.mm(A, A.t())
        B = AAT + torch.eye(self.M)
        LB = torch.cholesky(B)
        Aerr = torch.mm(A, y)
        c = torch.triangular_solve(Aerr, LB, upper=False)[0] / sigma

        K_ff = self.get_K_without_noise(z, z, x, x)
        Kdiag = K_ff.diag()

        bound = -0.5 * N * np.log(2 * np.pi) - torch.sum(torch.log(torch.diag(LB))) - 0.5 * N * torch.log(
            sigma2) - 0.5 * torch.sum(torch.pow(y, 2)) / sigma2
        bound += 0.5 * torch.sum(torch.pow(c, 2)) - 0.5 * torch.sum(Kdiag) / sigma2 + 0.5 * torch.sum(torch.diag(AAT))
        return bound

    def predict(self, z, x, y, z_star, x_star, add_likelihood_variance=False):
        subset = ~torch.isnan(y).reshape(-1)
        if subset.sum() > 0:
            y = y[subset, :]
            z = z[subset, :]
            x = x[subset, :]
        Nstar = z_star.size()[0]
        y = (y - self.intercept)
        sigma2 = self.get_noise_var()
        sigma = torch.sqrt(sigma2)

        K_all_inv = torch.inverse(
            self.get_K_without_noise(z, z, x, x, jitter=self.jitter) + sigma2 * torch.eye(y.size()[0]))

        K_uu = self.get_K_without_noise(self.z_u, self.z_u, self.x_u, self.x_u, jitter=self.jitter)
        K_uf = self.get_K_without_noise(self.z_u, z, self.x_u, x)
        K_us = self.get_K_without_noise(self.z_u, z_star, self.x_u, x_star)

        K_sf = self.get_K_without_noise(z_star, z, x_star, x)
        K_ss = self.get_K_without_noise(z_star, z_star, x_star, x_star)
        tmp = torch.mm(K_sf, K_all_inv)
        mean = self.intercept + torch.mm(tmp, y)
        var = torch.diag(K_ss - torch.mm(tmp, K_sf.t()))

        if add_likelihood_variance:
            var += self.get_noise_var()

        return mean.reshape(-1), var

    def log_prob_fullrank(self, z, x, y):
        N = z.size()[0]
        y = (y - self.intercept).reshape(-1)
        K = self.get_K_without_noise(z, z, x, x, jitter=self.jitter, which_kernels=None)
        K_noise = self.get_noise_var() * torch.eye(N)
        return MultivariateNormal(torch.zeros_like(y), K + K_noise).log_prob(y)

    def prior_loss(self):
        return -Gamma(1.0, 1.0).log_prob(self.get_kernel_var()).sum()

    def total_loss(self, z, x, y):
        return -self.log_prob(z, x, y) + self.prior_loss()


# GP with 1D inputs (useful for additive)
class GP_1D(nn.Module):

    def __init__(self, z_inducing):

        super().__init__()

        self.jitter = 1e-4

        # kernel hyperparameters
        self.ls = nn.Parameter(torch.ones(1), requires_grad=True)
        self.var = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.noise_var = nn.Parameter(-1.0 * torch.ones(1), requires_grad=True)
        self.intercept = nn.Parameter(torch.zeros(1), requires_grad=True)

        # inducing points
        self.z_u = z_inducing
        self.M = self.z_u.size()[0]

    def get_kernel_var(self):
        return 1e-4 + my_softplus(self.var)

    def get_ls(self):
        return 1e-4 + my_softplus(self.ls)

    def get_noise_var(self):
        return 1e-4 + my_softplus(self.noise_var)

    def kernel_decomposition(self, z, z2, jitter=None):
        ls = self.get_ls()
        var = self.get_kernel_var()
        K1 = RBF(z, z2, ls[0], var[0], jitter)
        return K1

    def get_K_without_noise(self, z, z2, jitter=None, which_kernels=None):
        K1 = self.kernel_decomposition(z, z2, jitter)
        return K1

    def log_prob(self, z, y):
        # inducing points log_prob
        subset = ~torch.isnan(y).reshape(-1)
        if subset.sum() > 0:
            y = y[subset, :]
            z = z[subset, :]
        N = y.size()[0]
        y = (y - self.intercept)
        sigma2 = self.get_noise_var()
        sigma = torch.sqrt(sigma2)
        K_uu = self.get_K_without_noise(self.z_u, self.z_u, jitter=self.jitter)
        K_uf = self.get_K_without_noise(self.z_u, z)
        L = torch.cholesky(K_uu)
        A = torch.triangular_solve(K_uf, L, upper=False)[0] / sigma
        AAT = torch.mm(A, A.t())
        B = AAT + torch.eye(self.M)
        LB = torch.cholesky(B)
        Aerr = torch.mm(A, y)
        c = torch.triangular_solve(Aerr, LB, upper=False)[0] / sigma

        K_ff = self.get_K_without_noise(z, z)
        Kdiag = K_ff.diag()

        bound = -0.5 * N * np.log(2 * np.pi) - torch.sum(torch.log(torch.diag(LB))) - 0.5 * N * torch.log(
            sigma2) - 0.5 * torch.sum(torch.pow(y, 2)) / sigma2
        bound += 0.5 * torch.sum(torch.pow(c, 2)) - 0.5 * torch.sum(Kdiag) / sigma2 + 0.5 * torch.sum(torch.diag(AAT))
        return bound

    def predict(self, z, y, z_star, which_kernels, add_likelihood_variance=False):
        subset = ~torch.isnan(y).reshape(-1)
        if subset.sum() > 0:
            y = y[subset, :]
            z = z[subset, :]
        Nstar = z_star.size()[0]
        y = (y - self.intercept)
        sigma2 = self.get_noise_var()
        sigma = torch.sqrt(sigma2)
        K_uu = self.get_K_without_noise(self.z_u, self.z_u, jitter=self.jitter, which_kernels=which_kernels)
        K_uf = self.get_K_without_noise(self.z_u, z, which_kernels=which_kernels)
        K_us = self.get_K_without_noise(self.z_u, z_star, which_kernels=which_kernels)

        L = torch.cholesky(K_uu)
        A = torch.triangular_solve(K_uf, L, upper=False)[0] / sigma
        AAT = torch.mm(A, A.t())
        B = AAT + torch.eye(self.M)
        LB = torch.cholesky(B)
        Aerr = torch.mm(A, y)
        c = torch.triangular_solve(Aerr, LB, upper=False)[0] / sigma

        tmp1 = torch.triangular_solve(K_us, L, upper=False)[0]
        tmp2 = torch.triangular_solve(tmp1, LB, upper=False)[0]
        mean = self.intercept + torch.mm(tmp2.t(), c)

        K_ss = self.get_K_without_noise(z_star, z_star)
        Kdiag = K_ss.diag()
        var = Kdiag + torch.pow(tmp2, 2).sum(dim=0) - torch.pow(tmp1, 2).sum(dim=0)

        if add_likelihood_variance:
            var += self.get_noise_var()

        return mean.reshape(-1), var

    def log_prob_fullrank(self, z, y):
        N = z.size()[0]
        y = (y - self.intercept).reshape(-1)
        K = self.get_K_without_noise(z, z, jitter=self.jitter, which_kernels=None)
        K_noise = self.get_noise_var() * torch.eye(N)
        return MultivariateNormal(torch.zeros_like(y), K + K_noise).log_prob(y)

    def prior_loss(self):
        return -Gamma(1.0, 1.0).log_prob(self.get_kernel_var()).sum()

    def total_loss(self, z, x, y):
        return -self.log_prob(z, x, y) + self.prior_loss()


class linear_mapping(nn.Module):

    def __init__(self):
        super(linear_mapping, self).__init__()

        self.jitter = 1e-3

        # parameters
        self.intercept = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(3), requires_grad=True)
        self.noise_var = nn.Parameter(-1.0 * torch.ones(1), requires_grad=True)

    def get_noise_var(self):
        return 1e-4 + my_softplus(self.noise_var)

    def get_kernel_var(self):
        return self.beta

    def log_prob(self, z, x, y):
        # inducing points log_prob
        N = y.size()[0]
        y_pred = self.intercept + self.beta[0] * z + self.beta[1] * x + self.beta[2] * z * x
        log_prob = Normal(y_pred, torch.sqrt(self.get_noise_var())).log_prob(y).sum()
        return log_prob

    def predict(self, z, x, y, z_star, x_star):
        mean = self.intercept + self.beta[0] * z_star + self.beta[1] * x_star + self.beta[2] * z_star * x_star
        return mean.reshape(-1), torch.zeros_like(mean).reshape(-1)

    def prior_loss(self):
        return -Gamma(1.0, 1.0).log_prob(self.beta).sum()

    def total_loss(self, z, x, y):
        return -self.log_prob(z, x, y)  # + self.prior_loss()
