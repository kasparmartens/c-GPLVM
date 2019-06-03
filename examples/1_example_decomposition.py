import torch
import pandas as pd

from cGPLVM.cGPLVM import cGPLVM
from cGPLVM.GP_mappings import GP_2D_AddInt
from cGPLVM.helpers import grid_helper

# generate toy data
def generate_toy_data(N, class_p=[1., 1., 1., 1., 1.], noise_sd=0.1):
    from torch.distributions.uniform import Uniform
    from torch.distributions.categorical import Categorical

    z = Uniform(-2.0, 2.0).rsample(sample_shape=(N, 1))
    x = Categorical(torch.Tensor(class_p)).sample(sample_shape=(N, 1)).float()-2
    z_positive = 0.5 + 0.5*(z > 0).float()
    y1 = torch.sin(z) + 0.2*x + 0.2*torch.sin(z)*x*z_positive + noise_sd*torch.randn(N, 1)
    Y = torch.cat([y1], dim=1)
    return z, x, Y

z, x, Y = generate_toy_data(N=1000)

# set up inducing points in the (z, x) space
z_inducing = torch.linspace(-2.0, 2.0, steps=10).reshape(-1, 1)
x_inducing = torch.linspace(-2.0, 2.0, steps=10).reshape(-1, 1)

# initialise covariate-GPLVM model
m = cGPLVM(x, Y, z, GP_mapping=GP_2D_AddInt, mean_zero=True, z_inducing=z_inducing, x_inducing=x_inducing, fixed_z=True)

# train the model using Adam
m.train(n_iter=1000, verbose=100)


### predictions

def helper_predict_decomposition(model,z_star, x_star):
    f_mean, _ = model.predict(z_star, x_star)
    f_z_mean, _ = model.predict_decomposition(z_star, x_star, which_kernels=[1.0, 0.0, 0.0])
    f_x_mean, _ = model.predict_decomposition(z_star, x_star, which_kernels=[0.0, 1.0, 0.0])
    f_int_mean, _ = model.predict_decomposition(z_star, x_star, which_kernels=[0.0, 0.0, 1.0])
    F_pred = torch.cat([z_star, x_star, f_mean, f_z_mean, f_x_mean, f_int_mean], dim=1)
    return F_pred

# create grid for predictions
grid = torch.linspace(-2.0, 2.0, steps=50).reshape(-1, 1)
z_grid, x_grid = grid_helper(grid, grid)

# predict the posterior mean
F_pred = helper_predict_decomposition(m, z_grid, x_grid).detach().numpy()

# write the predictions into csv file
col_names = ["z", "x", "f", "f_z", "f_x", "f_int"]
pd.DataFrame(F_pred, columns=col_names).to_csv("output/toy_decomposition.csv", index=False)

# plotting is done in R (see folder "plotting")
