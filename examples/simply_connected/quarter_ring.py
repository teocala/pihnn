"""
Test from Section 4.1.1 in https://doi.org/10.1016/j.cma.2024.117406
"""

import torch
import pihnn.nn as nn
import pihnn.utils as utils
import pihnn.geometries as geom
import pihnn.graphics as graphics
import pihnn.bc as bc

# Network parameters 
n_epochs = 2000 # Number of epochs
learn_rate = 1e-2 # Initial learning rate
scheduler_apply = [1500] # At which epoch to execute scheduler
units = [1, 10, 10, 10, 1] # Units in each network layer
np_train = 200 # Number of training points on domain boundary
np_test = 20 # Number of test points on the domain boundary
beta = 0.5 # Initialization parameter
gauss = 3 # Initalization parameter

# Geometry + boundary conditions
R = 2.0 # Domain radius
r = 0.5 # Internal radius
sig_ext = 1.0 # Normal tension applied on the external border
def bc_ext (x,y): return sig_ext*x/R, sig_ext*y/R

line1 = geom.line(P1=[0,r], P2=[0,R], bc_type=bc.symmetry_bc())
line2 = geom.line(P1=[-R,0], P2=[-r,0], bc_type=bc.symmetry_bc())
arc1 = geom.arc(center=[0,0], radius=r, theta1=torch.pi/2, theta2=torch.pi, bc_type=bc.stress_bc(), bc_value=0)
arc2 = geom.arc(center=[0,0], radius=R, theta1=torch.pi/2, theta2=torch.pi, bc_type=bc.stress_bc(), bc_value=bc_ext)
boundary = geom.boundary([line1, line2, arc1, arc2], np_train, np_test)

# Exact/Numerical solution, from Muskhelishvili 1954
def model_exact (x, y):
    z = x+1.j*y
    rho = torch.abs(z) 
    theta = torch.angle(z)
    pa = 0.0
    pb = - sig_ext
    sigma_rr = - 1.0/ (R**2 - r**2) * (pb*R**2 - pa*r**2 - (pb - pa)*(r*R/rho)**2)
    sigma_tt = - 1.0/ (R**2 - r**2) * (pb*R**2 - pa*r**2 + (pb - pa)*(r*R/rho)**2)
    sigma_xx = torch.cos(theta)**2 * sigma_rr + torch.sin(theta)**2 * sigma_tt
    sigma_yy = torch.sin(theta)**2 * sigma_rr + torch.cos(theta)**2 * sigma_tt 
    sigma_xy = torch.cos(theta) * torch.sin(theta) * (sigma_rr - sigma_tt)
    u_r = ((pa-pb)*r*r*R*R + 0.5*(r*r*pa-R*R*pb)*rho*rho)/(2*rho*(R*R-r*r))
    u_x = torch.cos(theta)*u_r
    u_y = torch.sin(theta)*u_r
    return torch.stack((sigma_xx, sigma_yy, sigma_xy, u_x, u_y))

# Definition of NN
model = nn.PIHNN('km', units)

if (__name__=='__main__'):
    model.initialize_weights('exp', beta, boundary.extract_points(10*boundary.np_train)[0], gauss)
    graphics.plot_training_points(boundary)
    loss_train, loss_test = utils.train(boundary, model, n_epochs, learn_rate, scheduler_apply)
    graphics.plot_loss(loss_train, loss_test)
    tria = graphics.get_triangulation(boundary)
    graphics.plot_sol(tria, model, model_exact)