import math
import torch

from torch.distributions.weibull import Weibull
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal

def Phi(x):
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def Phi_inverse(value):
    return torch.erfinv(2 * value - 1) * math.sqrt(2)


def rsample_TruncatedNormal(loc, scale, lower, upper, sample_shape=(1,)):
    F_a = Phi((lower - loc) / scale)
    F_b = Phi((upper - loc) / scale)
    u0 = Uniform(torch.zeros_like(loc), torch.ones_like(scale)).rsample(sample_shape=sample_shape)
    u = u0 * (F_b - F_a) + F_a
    out = Phi_inverse(u) * scale + loc
    if torch.isinf(out).any():
        print("rsample_TruncatedNormal has a problem")
    return Phi_inverse(u) * scale + loc


def log_prob_Normal(value, loc, scale):
    return Normal(loc=loc, scale=scale).log_prob(value)


def log_prob_TruncatedNormal(value, loc, scale, lower, upper):
    F_a = Phi(lower)
    F_b = Phi(upper)
    return Normal(loc=loc, scale=scale).log_prob(value) - torch.log(F_b - F_a + 1e-8)


def log_prob_Weibull(x, shape, scale):
    x_over_lambda = x / scale
    return torch.log(shape / scale) + (shape - 1) * torch.log(x_over_lambda) - torch.pow(x_over_lambda, shape)


def log_prob_TruncatedWeibull(x, shape, scale, lower, upper):
    p = Weibull(scale=scale, concentration=shape)
    F_a = p.cdf(lower)
    F_b = p.cdf(upper)
    return log_prob_Weibull(x, shape, scale) - torch.log(F_b - F_a + 1e-8)


def inv_cdf_Weibull(u, shape, scale):
    # inverse_cdf(u) = - lambda * log(1-u)^{1/k}
    return scale * torch.pow(-torch.log(1.0 - u), 1.0 / shape)


# Truncated Weibull(lambda, k), [lower, upper]
def rsample_TruncatedWeibull(shape, scale, lower, upper, sample_shape=(1,)):
    p = Weibull(scale=scale, concentration=shape)
    F_a = p.cdf(lower)
    F_b = p.cdf(upper)
    # u0 = Uniform(0.0, 1.0).rsample(sample_shape=sample_shape)
    u0 = Uniform(torch.zeros_like(scale), torch.ones_like(scale)).rsample(sample_shape=sample_shape)
    u = u0 * (F_b - F_a + 1e-12) + F_a
    return inv_cdf_Weibull(u, shape, scale)

def calculate_KLqp_TruncatedWeibullNormal(p_shape, p_scale, q_loc, q_scale, lower, upper, n_samples=20):
    # sample from q
    sample_q = rsample_TruncatedNormal(q_loc, q_scale, lower, upper, sample_shape=(n_samples,))
    # evaluate log_probs
    logp = torch.zeros_like(q_loc)  # log_prob_TruncatedWeibull(sample_q, p_shape, p_scale, lower, upper)
    logq = log_prob_TruncatedNormal(sample_q, q_loc, q_scale, lower, upper)
    # set NaNs to zero
    logp[torch.isnan(logp)] = 0.0
    # print((sample_q.min().data, sample_q.max().data, logp.sum().data, logq.sum().data))
    # average across replicates, sum over data points
    KL_avg = torch.mean(logq - logp, dim=0)
    KL = torch.sum(KL_avg)
    return KL
