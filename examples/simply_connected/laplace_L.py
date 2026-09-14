"""
Test 3.1 in https://doi.org/10.1007/s44379-026-00094-8 with PIHKANs
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
n_epochs = 2000 # Number of epochs
learn_rate = 5e-3 # Initial learning rate 4e-3 for units=12
scheduler_apply = [1500] # At which epoch to execute scheduler
units = [1,10,10,10,10,10,1] # Units in each network layer
degree = 5 # KAN polynomial degree
np_train = 800 # Number of training points on domain boundary
np_test = 100 # Number of test points on the domain boundary
beta = 0.5 # Initialization parameter
gauss = 3 # Initalization parameter

# Geometry + boundary conditions
L = 1 # Domain half length

# Exact/Numerical solution
dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
x_FE, y_FE, u_FE = np.loadtxt(dir_path+"/FEM_solutions/laplace_L.rpt").T
u_FE = scipy.interpolate.NearestNDInterpolator(list(zip(x_FE,y_FE)), u_FE)
def model_FE(x,y): return torch.tensor(u_FE(x,y).T)

line1 = geom.line(P1=[-L,-L], P2=[-L,L], bc_type=bc.dirichlet_bc(), bc_value=0)
line2 = geom.line(P1=[-L,L], P2=[0,L], bc_type=bc.dirichlet_bc(), bc_value=0)
line3 = geom.line(P1=[0,L], P2=[0,0], bc_type=bc.dirichlet_bc(), bc_value=0)
line4 = geom.line(P1=[0,0], P2=[L,0], bc_type=bc.dirichlet_bc(), bc_value=0)
line5 = geom.line(P1=[L,0], P2=[L,-L], bc_type=bc.dirichlet_bc(), bc_value=0)
line6 = geom.line(P1=[L,-L], P2=[-L,-L], bc_type=bc.dirichlet_bc(), bc_value=0)
boundary = geom.boundary([line1, line2, line3, line4, line5, line6], np_train, np_test)

# Definition of NN
rhs_solution = lambda x,y: -x*x/4 - y*y/4
model = nn.PIHKAN('laplace', units, degree, rhs_solution=rhs_solution)

if (__name__=='__main__'):
    model.initialize_weights('kan', sample=boundary.extract_points(10*np_train)[0])
    graphics.plot_training_points(boundary)
    loss_train, loss_test = utils.train(boundary, model, n_epochs, learn_rate, scheduler_apply)
    graphics.plot_loss(loss_train, loss_test)
    tria = graphics.get_triangulation(boundary)
    graphics.plot_sol(tria, model, model_FE)