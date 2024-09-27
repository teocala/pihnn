"""
Resolution of the Laplace problem on a square with mixed boundary conditions (simply_connected/laplace.py with DD-PIHNNs).
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
def u_exact (x,y): return x*x - y*y

# Geometry + boundary conditions
L = 0.5 # Domain half length
def bc_R (x,y): return 2*x # From u_exact
def bc_L (x,y): return -2*x
def bc_B (x,y): return 2*y

line1 = geom.line(P1=[L,L], P2=[0,L], bc_type=bc.dirichlet_bc(), bc_value=u_exact)
line2 = geom.line(P1=[0,L], P2=[-L,L], bc_type=bc.dirichlet_bc(), bc_value=u_exact)
line3 = geom.line(P1=[-L,L], P2=[-L,-L], bc_type=bc.neumann_bc(), bc_value=bc_L)
line4 = geom.line(P1=[-L,-L], P2=[0,-L], bc_type=bc.neumann_bc(), bc_value=bc_B)
line5 = geom.line(P1=[0,-L], P2=[L,-L], bc_type=bc.neumann_bc(), bc_value=bc_B)
line6 = geom.line(P1=[L,-L], P2=[L,L], bc_type=bc.neumann_bc(), bc_value=bc_R)
line7 = geom.line(P1=[0,-L], P2=[0,L], bc_type=bc.interface_bc())

def dd_partition (x,y): # Domain decomposition partition
    domains = torch.empty([2,x.shape[0]], dtype=torch.bool)
    domains[0,:] = (x>=-1e-10)
    domains[1,:] = (x<=1e-10)
    return domains
boundary = geom.boundary([line1, line2, line3, line4, line5, line6, line7], np_train, np_test, dd_partition)

# Definition of NN
model = nn.DD_PIHNN('laplace', units, boundary).to(nn.device)

if (__name__=='__main__'):
    model.initialize_weights('exp', beta, boundary.extract_points(10*np_train)[0], gauss)
    #graphics.plot_training_points(boundary)
    loss_train, loss_test = utils.train(boundary, model, n_epochs, learn_rate, scheduler_apply)
    graphics.plot_loss(loss_train, loss_test)
    tria = graphics.get_triangulation(boundary)
    graphics.plot_sol(tria, model, u_exact)