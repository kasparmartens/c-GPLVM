import torch
from torch.distributions.normal import Normal

# define KL between normal and standard normal distribution
def KL_standard_normal(mu, sigma):
    p = Normal(torch.zeros_like(mu), torch.ones_like(mu))
    q = Normal(mu, sigma)
    return torch.sum(torch.distributions.kl_divergence(q, p))


# define softplus function
def my_softplus(x):
    return torch.log(1.0 + torch.exp(x))

def create_2D_grid(grid_x1, grid_x2, device="cpu"):
    x1_s, x2_s = torch.meshgrid([grid_x1.to(device), grid_x2.to(device)])
    x1_star, x2_star = x1_s.reshape([-1, 1]), x2_s.reshape([-1, 1])
    return x1_star, x2_star

# takes in [N1, 1] and [N2, p2] matrices, then performs expand grid
def grid_helper(a, b):
    nrow_a = a.size()[0]
    nrow_b = b.size()[0]
    ncol_b = b.size()[1]
    x = a.repeat(nrow_b, 1)
    y = b.repeat(1, nrow_a).view(-1, ncol_b)
    return x, y
