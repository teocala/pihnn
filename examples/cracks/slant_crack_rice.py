"""
Test from Section 4.2 in https://doi.org/10.1016/j.engfracmech.2025.111133
"""

import os
import torch
import scipy
import pihnn.nn as nn
import pihnn.utils as utils
import pihnn.geometries as geom
import pihnn.bc as bc
import pihnn.graphics as graphics
import numpy as np

# Network parameters 
n_epochs = 3000 # Number of epochs
learn_rate = 1e-2 # Initial learning rate
scheduler_apply = [1000, 2000, 2900] # At which epoch to execute scheduler
units = [1, 10, 10, 10, 1] # Units in each network layer
np_train = 1000 # Number of training points on domain boundary
np_test = 40 # Number of test points on the domain boundary
beta = 0.5 # Initialization parameter
gauss = 3 # Initalization parameter

# Geometry + boundary conditions
h = 5 # Domain half height
l = 5 # Domain half length
a = 1.5 # Crack half lengths
alpha = torch.pi/6 # Crack angle
sig_ext_t = lambda z: z.real*z.real*1j # Normal tension applied above
sig_ext_b = lambda z: -z.real*z.real*1j # Normal tension applied below

line1 = geom.line(P1=[-l,-h], P2=[l,-h], bc_type=bc.stress_bc(), bc_value=sig_ext_b)
line2 = geom.line(P1=[l,-h], P2=[l,h], bc_type=bc.stress_bc(), bc_value=0)
line3 = geom.line(P1=[l,h], P2=[-l,h], bc_type=bc.stress_bc(), bc_value=sig_ext_t)
line4 = geom.line(P1=[-l,h], P2=[-l,-h], bc_type=bc.stress_bc(), bc_value=0)
crack = geom.line(P1=-a*np.exp(1j*alpha)-1j+1, P2=a*np.exp(1j*alpha)-1j+1, bc_type=bc.stress_bc())

crack.add_crack_tip(tip_side=0)
crack.add_crack_tip(tip_side=1)
boundary = geom.boundary([line1, line2, line3, line4, crack], np_train, np_test, enrichment='rice')

# Exact/Numerical solution
dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
x_FE, y_FE, sig_xx_FE, sig_yy_FE, sig_xy_FE = np.loadtxt(dir_path+"/FEM_solutions/slant_crack.rpt").T
vars_FE = np.stack((sig_xx_FE, sig_yy_FE, sig_xy_FE), axis=1)
vars_FE = scipy.interpolate.NearestNDInterpolator(list(zip(x_FE,y_FE)), vars_FE)
def model_FE(x,y): return torch.tensor(vars_FE(x,y).T)

# Definition of NN
model = nn.enriched_PIHNN('km', units, boundary)

if (__name__=='__main__'):
    model.initialize_weights('exp', beta, boundary.extract_points(10*np_train)[0], gauss)
    loss_train, loss_test = utils.train(boundary, model, n_epochs, learn_rate, scheduler_apply)
    graphics.plot_loss(loss_train, loss_test)
    tria = graphics.get_triangulation(boundary)
    graphics.plot_sol(tria, model, model_FE, apply_crack_bounds=True) # We bound the crack singularities for the plot

    sif_analytical = torch.tensor(12.555+2.613*1j)
    sif_i_yw, sif_ii_yw = utils.yau_wang_method(model, crack.tips[1], a)
    sif_yw = sif_i_yw + 1j*sif_ii_yw
    err = torch.abs(sif_analytical-sif_yw)/torch.abs(sif_analytical)

    print(f"Theoretical SIF: {sif_analytical:.3f}")
    print(f"SIFs from J-integral: {sif_yw:.3f}")
    print(f"Relative SIF error: {err:.1e}")