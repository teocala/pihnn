"""
Test from Section 4.2.3 in https://doi.org/10.1016/j.cma.2024.117406
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
n_epochs = 40000 # Number of epochs
learn_rate = 2e-4 # Initial learning rate
scheduler_apply = [] # At which epoch to execute scheduler
units = [1, 30, 30, 30, 30, 30, 1] # Units in each network layer
np_train = 400 # Number of training points on domain boundary
np_test = 40 # Number of test points on the domain boundary
beta = 0.5 # Initialization parameter
gauss = 3 # Initalization parameter

# Geometry + boundary conditions
a45 = torch.pi/4
v0 = torch.tensor([1,0])
v90 = torch.tensor([0,1])
v45 = torch.tensor([np.sqrt(2)/2,np.sqrt(2)/2])
v135 = torch.tensor([-v45[0],v45[1]])

c1 = torch.tensor([-3,4])
z1 = c1 + 3*v45
z2 = z1 + 2*v135
c2 = z2 + 3*v45
z3 = c2 - 3*v0
c3 = z3 + 8*v0
c4 = c3 - 6*v135
z4 = c4 - 3*v0
c5 = z4 - 0.41*v90 + 0.5*v0
z5 = c5 - 0.5*v90

bc_top = [0,-1]

line1 = geom.line(10, 10+4.5*1.j, bc_type=bc.stress_bc())
line2 = geom.line(10+4.5*1.j, z5, bc_type=bc.stress_bc())
arc1 = geom.arc(c5, 0.5, 6*a45, 4*a45, bc_type=bc.stress_bc())
line3 = geom.line(z4-0.41*v90, z4, bc_type=bc.stress_bc())
arc2 = geom.arc(c4, 3, 4*a45, 3*a45, bc_type=bc.stress_bc())
arc3 = geom.arc(c3, 3, -a45, 0, bc_type=bc.stress_bc())
line4 = geom.line(z3+11*v0, z3+2*v90+11*v0, bc_type=bc.stress_bc())
line5 = geom.line(z3+2*v90+11*v0, z3+2*v90, bc_type=bc.stress_bc(), bc_value=bc_top) # Top
line6 = geom.line(z3+2*v90, z3, bc_type=bc.stress_bc())
arc4 = geom.arc(c2, 3, 4*a45, 5*a45, bc_type=bc.stress_bc())
line7 = geom.line(z2, z1, bc_type=bc.stress_bc())
arc5 = geom.arc(c1, 3, a45, 0, bc_type=bc.stress_bc())
line8 = geom.line(4*v90, 0, bc_type=bc.stress_bc())
line9 = geom.line(0, 10, bc_type=bc.displacement_bc()) # Bottom

boundary = geom.boundary([line1, line2, line3, line4, line5, line6, line7, line8, line9, arc1, arc2, arc3, arc4, arc5], np_train, np_test)

# Exact/Numerical solution
dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
x_FE, y_FE, sig_xx_FE, sig_yy_FE, sig_xy_FE, u_x_FE, u_y_FE = np.loadtxt(dir_path+"/FEM_solutions/rail_section.rpt").T
vars_FE = np.stack((sig_xx_FE, sig_yy_FE, sig_xy_FE, u_x_FE, u_y_FE), axis=1)
vars_FE = scipy.interpolate.NearestNDInterpolator(list(zip(x_FE,y_FE)), vars_FE)
def model_FE(x,y): return torch.tensor(vars_FE(x,y).T)

# Definition of NN
model = nn.PIHNN('km', units).to(nn.device)

if (__name__=="__main__"):
    model.initialize_weights('exp', beta, boundary.extract_points(10*np_train)[0], gauss)
    graphics.plot_training_points(boundary)
    loss_train, loss_test = utils.train(boundary, model, n_epochs, learn_rate, scheduler_apply)
    graphics.plot_loss(loss_train, loss_test)
    tria = graphics.get_triangulation(boundary)
    graphics.plot_sol(tria, model, model_FE)