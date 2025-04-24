"""
Test from Section 4.1 in https://doi.org/10.1016/j.engfracmech.2025.111133
"""

import torch
import pihnn.nn as nn
import pihnn.utils as utils
import pihnn.geometries as geom
import pihnn.bc as bc
import pihnn.graphics as graphics

# Network parameters 
n_epochs = 1000 # Number of epochs
learn_rate = 1e-2 # Initial learning rate
scheduler_apply = [] # At which epoch to execute scheduler
units = [1, 10, 10, 10, 1] # Units in each network layer
np_train = 1000 # Number of training points on domain boundary
np_test = 40 # Number of test points on the domain boundary
beta = 0.5 # Initialization parameter
gauss = 3 # Initalization parameter

# Geometry + boundary conditions
h = 6 # Domain half height
l = 4 # Domain half length
a = 1 # Crack half length
sig_ext = 1 # Normal tension applied on the external border

line1 = geom.line(P1=[-l,-h], P2=[l,-h], bc_type=bc.stress_bc(), bc_value=-sig_ext*1j)
line2 = geom.line(P1=[l,-h], P2=[l,h], bc_type=bc.stress_bc(), bc_value=0)
line3 = geom.line(P1=[l,h], P2=[-l,h], bc_type=bc.stress_bc(), bc_value=sig_ext*1j)
line4 = geom.line(P1=[-l,h], P2=[-l,-h], bc_type=bc.stress_bc(), bc_value=0)
crack = geom.line(P1=[-a,0], P2=[a,0], bc_type=bc.stress_bc(), bc_value=0.)
interface = geom.line(P1=[a,0],P2=[l,0],bc_type=bc.interface_bc())
interface2 = geom.line(P1=[-a,0],P2=[-l,0],bc_type=bc.interface_bc())

def dd_partition (x,y): # Domain decomposition partition
    return torch.stack([
    (y>=-1e-10),
    (y<=1e-10)
    ])

crack.add_crack_tip(tip_side=0)
crack.add_crack_tip(tip_side=1)
boundary = geom.boundary([line1, line2, line3, line4, crack, interface, interface2], np_train, np_test, dd_partition, enrichment='williams')

# Definition of NN
model = nn.enriched_PIHNN('km', units, boundary)

if (__name__=='__main__'):
    graphics.plot_training_points(boundary)
    model.initialize_weights('exp', beta, boundary.extract_points(10*np_train)[0], gauss)
    loss_train, loss_test = utils.train(boundary, model, n_epochs, learn_rate, scheduler_apply)
    tria = graphics.get_triangulation(boundary)
    graphics.plot_sol(tria, model)

    sif_analytical = torch.tensor(2.103) if l==2 else torch.tensor(1.852)
    sif_i_yw, sif_ii_yw = utils.yau_wang_method(model, crack.tips[0], a)
    sif_yw = sif_i_yw + 1j*sif_ii_yw
    err = torch.abs(sif_analytical-sif_yw)/torch.abs(sif_analytical)

    print(f"Theoretical SIF: {sif_analytical:.3f}")
    print(f"SIFs from J-integral: {sif_yw:.3f}")
    print(f"Relative SIF error: {err:.1e}")