import math
import torch

def RBF(x, y, lengthscale, variance, jitter=None):
    N = x.size()[0]
    x = x / lengthscale
    y = y / lengthscale
    s_x = torch.sum(torch.pow(x, 2), dim=1).reshape([-1, 1])
    s_y = torch.sum(torch.pow(y, 2), dim=1).reshape([1, -1])
    K = variance * torch.exp(- 0.5 * (s_x + s_y - 2 * torch.mm(x, y.t())))
    if jitter is not None:
        K += jitter * torch.eye(N)
    return K


def meanzeroRBF(x, y, lengthscale, variance, a, b, jitter=None):
    N = x.size()[0]
    if x.size()[1] != 1:
        raise ValueError("Not implemented for input dim > 1")

    sqrt2 = math.sqrt(2)
    sqrtpi = math.sqrt(math.pi)
    sqrt2lengthscale = sqrt2 * lengthscale
    K = RBF(x, y, lengthscale, variance)

    const = 0.5 * sqrtpi * sqrt2lengthscale * variance
    K12 = (torch.erf((b - x) / sqrt2lengthscale) - torch.erf((a - x) / sqrt2lengthscale))
    K21T = (torch.erf((b - y) / sqrt2lengthscale) - torch.erf((a - y) / sqrt2lengthscale))

    temp = (b - a) / (sqrt2lengthscale)
    K22 = 2 * ((a - b) * torch.erf((a - b) / sqrt2lengthscale * torch.ones(1)) + sqrt2 / sqrtpi * lengthscale * (
                torch.exp(-temp ** 2) - 1))
    out = K - const * torch.mm(K12, K21T.t()) / K22
    if jitter is not None:
        out += jitter * torch.eye(N)
    return out


# calculates the diagonal of mean-zero kernel matrix
def diag_meanzeroRBF(x, lengthscale, variance, a, b, jitter=None):
    N = x.size()[0]
    x = x.reshape(-1)

    sqrt2 = math.sqrt(2)
    sqrtpi = math.sqrt(math.pi)
    sqrt2lengthscale = sqrt2 * lengthscale
    Kdiag = variance * torch.ones(N)

    const = 0.5 * sqrtpi * sqrt2lengthscale * variance
    K12 = (torch.erf((b - x) / sqrt2lengthscale) - torch.erf((a - x) / sqrt2lengthscale))

    temp = (b - a) / (sqrt2lengthscale)
    K22 = 2 * ((a - b) * torch.erf((a - b) / sqrt2lengthscale * torch.ones(1)) + sqrt2 / sqrtpi * lengthscale * (
                torch.exp(-temp ** 2) - 1))
    out = Kdiag - const * K12 * K12 / K22
    return out


def addint_2D_kernel_decomposition(z, z2, x, x2, ls, var, a=-3, b=3, mean_zero=True, jitter=None):
    if mean_zero:
        K_zz = meanzeroRBF(z, z2, ls[0], var[0], a, b, jitter)

        K_xx = meanzeroRBF(x, x2, ls[1], var[1], a, b, jitter)

        K_intz = meanzeroRBF(z, z2, ls[2], var[2], a, b, jitter)
        K_intx = meanzeroRBF(x, x2, ls[3], 1.0, a, b, jitter)
        K_int = K_intz * K_intx
    else:
        K_zz = RBF(z, z2, ls[0], var[0], jitter)

        K_xx = RBF(x, x2, ls[1], var[1], jitter)

        K_intz = RBF(z, z2, ls[2], var[2], jitter)
        K_intx = RBF(x, x2, ls[3], 1.0, jitter)
        K_int = K_intz * K_intx

    return K_zz, K_xx, K_int


def addint_kernel_diag(z, x, ls, var, a=-3, b=3, mean_zero=True, jitter=None):
    P = x.shape[1]
    if mean_zero:
        # f(z)
        K_zz = diag_meanzeroRBF(z, ls[0], var[0], a, b, jitter)
        # f(x)
        K_xx = sum([diag_meanzeroRBF(x[:, j:(j + 1)], ls[1 + j], var[1 + j], a, b, jitter) for j in range(P)])

        # f(x, z)
        def productkernel(j, z, x, ls, var, a, b, jitter):
            K_intz = diag_meanzeroRBF(z, ls[1 + j + P], var[1 + j + P], a, b, jitter)
            K_intx = diag_meanzeroRBF(x[:, j:(j + 1)], ls[1 + j + 2 * P], 1.0, a, b, jitter)
            return K_intz * K_intx

        K_int = sum([productkernel(j, z, x, ls, var, a, b, jitter) for j in range(P)])
    else:
        raise ValueError("not implemented")

    return K_zz, K_xx, K_int
