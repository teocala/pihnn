"""
Test from Section 4.3 in https://arxiv.org/abs/2407.01088
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
n_epochs = 3000 # Number of epochs
learn_rate = 1e-2 # Initial learning rate
scheduler_apply = [500, 1000, 2500] # At which epoch to execute scheduler
units = [1, 10, 10, 10, 1] # Units in each network layer
np_train = 600 # Number of training points on domain boundary
np_test = 60 # Number of test points on the domain boundary
beta = 0.5 # Initialization parameter
gauss = 3 # Initalization parameter

# Exact/Numerical solution
dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
x_FE, y_FE, sig_xx_FE, sig_yy_FE, sig_xy_FE, u_x_FE, u_y_FE = np.loadtxt(dir_path+"/FEM_solutions/plate_hole.rpt").T
vars_FE = np.stack((sig_xx_FE, sig_yy_FE, sig_xy_FE, u_x_FE, u_y_FE), axis=1)
vars_FE = scipy.interpolate.LinearNDInterpolator(list(zip(x_FE,y_FE)), vars_FE)
def model_FE(x,y): return torch.tensor(vars_FE(x,y).T)
def disp_FE(x,y): return torch.tensor(vars_FE(x,y).T[3:4])

# Geometry + boundary conditions
L = 2.5 # Domain square half length
r = 1 # Internal radius
sig_ext = 1.0 # Normal tension applied on the external border
line1 = geom.line(P1=[L,L], P2=[0,L], bc_type=bc.stress_bc(), bc_value=0) # top
line2 = geom.line(P1=[0,L], P2=[-L,L], bc_type=bc.stress_bc(), bc_value=0) # top
line3 = geom.line(P1=[-L,L], P2=[-L,0], bc_type=bc.stress_bc(), bc_value=-sig_ext) # left
line4 = geom.line(P1=[-L,0], P2=[-L,-L], bc_type=bc.stress_bc(), bc_value=-sig_ext) # left
line5 = geom.line(P1=[-L,-L], P2=[0,-L], bc_type=bc.stress_bc(), bc_value=0) # bottom
line6 = geom.line(P1=[0,-L], P2=[L,-L], bc_type=bc.stress_bc(), bc_value=0) # bottom
line7 = geom.line(P1=[L,-L], P2=[L,0], bc_type=bc.stress_bc(), bc_value=sig_ext) # right
line8 = geom.line(P1=[L,0], P2=[L,L], bc_type=bc.stress_bc(), bc_value=sig_ext) # right
arc1 = geom.arc(center=[0,0], radius=r, theta1=0, theta2=torch.pi/2, bc_type=bc.stress_bc(), bc_value=0)
arc2 = geom.arc(center=[0,0], radius=r, theta1=torch.pi/2, theta2=torch.pi, bc_type=bc.stress_bc(), bc_value=0)
arc3 = geom.arc(center=[0,0], radius=r, theta1=torch.pi, theta2=3*torch.pi/2, bc_type=bc.stress_bc(), bc_value=0)
arc4 = geom.arc(center=[0,0], radius=r, theta1=3*torch.pi/2, theta2=2*torch.pi, bc_type=bc.stress_bc(), bc_value=0)
line9 = geom.line(P1=[-L,0], P2=[-r,0], bc_type=bc.interface_bc())
line10 = geom.line(P1=[r,0], P2=[L,0], bc_type=bc.interface_bc())
line11 = geom.line(P1=[0,-L], P2=[0,-r], bc_type=bc.interface_bc())
line12 = geom.line(P1=[0,r], P2=[0,L], bc_type=bc.interface_bc())
line13 = geom.line(P1=[0,r], P2=[0,L], bc_type=bc.normal_displacement_bc(), on_boundary=False) # To avoid rotations, guaranteeing uniqueness of displacements
line14 = geom.line(P1=[-L,0], P2=[-r,0], bc_type=bc.normal_displacement_bc(), on_boundary=False)

def dd_partition (x,y): # Domain decomposition partition
    domains = torch.empty([4,x.shape[0]], dtype=torch.bool)
    domains[0,:] = (x>=-1e-10) & (y>=-1e-10)
    domains[1,:] = (x<=1e-10) & (y>=-1e-10)
    domains[2,:] = (x<=1e-10) & (y<=1e-10)
    domains[3,:] = (x>=-1e-10) & (y<=1e-10)
    return domains
boundary = geom.boundary([line1, line2, line3, line4, line5, line6, line7, line8, line9, line10, line11, line12, line13, line14, arc1, arc2, arc3, arc4], np_train, np_test, dd_partition)

# Definition of NN
model = nn.DD_PIHNN('km', units, boundary).to(nn.device)

if (__name__=='__main__'):
    model.initialize_weights('exp', beta, boundary.extract_points(10*np_train)[0], gauss)
    graphics.plot_training_points(boundary)
    loss_train, loss_test = utils.train(boundary, model, n_epochs, learn_rate, scheduler_apply)
    graphics.plot_loss(loss_train, loss_test)
    tria = graphics.get_triangulation(boundary)
    graphics.plot_sol(tria, model, model_FE)