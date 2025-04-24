"""
Resolution of the Laplace problem on a square with mixed boundary conditions.
"""

import torch
import pihnn.nn as nn
import pihnn.utils as utils
import pihnn.geometries as geom
import pihnn.graphics as graphics
import pihnn.bc as bc

# Network parameters 
n_epochs = 800 # Number of epochs
learn_rate = 3e-2 # Initial learning rate
scheduler_apply = [] # At which epoch to execute scheduler
units = [1, 10, 10, 1] # Units in each network layer
np_train = 200 # Number of training points on domain boundary
np_test = 20 # Number of test points on the domain boundary
beta = 0.5 # Initialization parameter
gauss = 3 # Initalization parameter

# Exact solution 
def u_exact (x,y): return x*x - y*y + torch.exp(x)*torch.cos(y)

# Geometry + boundary conditions
L = 0.5 # Domain half length
def bc_R (x,y): return 2*x + torch.exp(x)*torch.cos(y) # From u_exact
def bc_L (x,y): return -2*x - torch.exp(x)*torch.cos(y)
def bc_B (x,y): return 2*y + torch.exp(x)*torch.sin(y)

line1 = geom.line(P1=[L,L], P2=[-L,L], bc_type=bc.dirichlet_bc(), bc_value=u_exact)
line2 = geom.line(P1=[-L,L], P2=[-L,-L], bc_type=bc.neumann_bc(), bc_value=bc_L)
line3 = geom.line(P1=[-L,-L], P2=[L,-L], bc_type=bc.neumann_bc(), bc_value=bc_B)
line4 = geom.line(P1=[L,-L], P2=[L,L], bc_type=bc.neumann_bc(), bc_value=bc_R)
boundary = geom.boundary([line1, line2, line3, line4], np_train, np_test)

# Definition of NN
model = nn.PIHNN('laplace', units)

if (__name__=='__main__'):
    model.initialize_weights('exp', beta, boundary.extract_points(10*boundary.np_train)[0], gauss)
    graphics.plot_training_points(boundary)
    loss_train, loss_test = utils.train(boundary, model, n_epochs, learn_rate, scheduler_apply)
    graphics.plot_loss(loss_train, loss_test)
    tria = graphics.get_triangulation(boundary)
    graphics.plot_sol(tria, model, u_exact)