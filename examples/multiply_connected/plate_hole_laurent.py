"""
Same test as in plate_hole_dd.py except the use of L-PIHNNs.
"""

import os
import torch
import numpy as np
import scipy
import pihnn.nn as nn
import pihnn.utils as utils
import pihnn.geometries as geom
import pihnn.graphics as graphics
import pihnn.bc as bc

# Network parameters 
n_epochs = 3500 # Number of epochs
learn_rate = 1e-2 # Initial learning rate
scheduler_apply = [1000, 2500] # At which epoch to execute scheduler
units = [1, 30, 30, 30, 1] # Units in each network layer
#units = [1, 10, 10, 10, 10, 1] # Units in each network layer
degree = 3 # KAN polynomial degree
units_s = [1, 10, 10, 1] # Units in each network layer in each singularity branch
location_s = 0.j # Location of each singularity
np_train = 200 # Number of training points on domain boundary
np_test = 20 # Number of test points on the domain boundary
beta = 0.5 # Initialization parameter
gauss = 3 # Initalization parameter

# Geometry + boundary conditions
L = 2.5 # Domain square half length
r = 1 # Internal radius
sig_ext = 1.0 # Normal tension applied on the external border
line1 = geom.line(P1=[L,L], P2=[-L,L], bc_type=bc.stress_bc(), bc_value=0)
line2 = geom.line(P1=[-L,L], P2=[-L,-L], bc_type=bc.stress_bc(), bc_value=-sig_ext)
line3 = geom.line(P1=[-L,-L], P2=[L,-L], bc_type=bc.stress_bc(), bc_value=0)
line4 = geom.line(P1=[L,-L], P2=[L,L], bc_type=bc.stress_bc(), bc_value=sig_ext)
circle = geom.circle(center=0, radius=r, bc_type=bc.stress_bc(), bc_value=0)
boundary = geom.boundary([line1, line2, line3, line4, circle], np_train, np_test)


# Exact/Numerical solution
dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
x_FE, y_FE, sig_xx_FE, sig_yy_FE, sig_xy_FE, u_x_FE, v_y_FE = np.loadtxt(dir_path+"/FEM_solutions/plate_hole.rpt").T
vars_FE = np.stack((sig_xx_FE, sig_yy_FE, sig_xy_FE), axis=1)
vars_FE = scipy.interpolate.LinearNDInterpolator(list(zip(x_FE,y_FE)), vars_FE)
def model_FE(x,y): return torch.tensor(vars_FE(x,y).T)


singularity = {
  "location": location_s,
  "degree": degree,
  "units": units_s
}

# Definition of NN
model = nn.L_PIHNN('km', units, [singularity]).to(nn.device)
#model = nn.LPIHKAN('km', units, degree, [singularity]).to(nn.device)

if (__name__=='__main__'):
    graphics.plot_training_points(boundary)
    model.initialize_weights('exp', beta, boundary.extract_points(10*np_train)[0], gauss)
    #model.initialize_normalization(boundary.extract_points(10*np_train)[0])
    loss_train, loss_test = utils.train(boundary, model, n_epochs, learn_rate, scheduler_apply)
    graphics.plot_loss(loss_train, loss_test)
    tria = graphics.get_triangulation(boundary)
    graphics.plot_sol(tria, model, model_FE)