"""Utilities for building and training PIHNNs."""
import torch
import pihnn.nn as nn
import pihnn.bc as bc
import numpy as np
from tqdm import tqdm
import os, inspect, warnings


def ordinal_number(n):
    """
    Accessory function to get the ordinal number as a string from an int.
    For example, :math:`3\\rightarrow \\text{3rd}`, :math:`10\\rightarrow \\text{10th}`, ...

    :param n: Input integer.
    :type n: int
    :returns: 
        - **ord_n** (string) - Ordinal number.
    """
    return "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4]) # From https://codegolf.stackexchange.com/questions/4707/outputting-ordinal-numbers-1st-2nd-3rd#answer-4712


def get_complex_input(input_value):
    """
    | Convert the input value to the unified format (i.e., complex :class:`torch.tensor`).
    | The library allows to define some values in multiple and flexible ways (scalars, tuples, lists, tensors). 
      Then, this method includes all the possible definitions and unify the type of the input value.

    :param input_value: Generic input value.
    :type input_value: int/float/complex/list/tuple/:class:`torch.tensor`
    :returns: 
        - **new_value** (callable) - Copy of the input value in the unified format.
    """
    if isinstance(input_value, (int,float,complex)):
        return torch.tensor(input_value +0j, device=nn.device)
    elif isinstance(input_value, (list,tuple)):
        if len(input_value) == 1:
            return torch.tensor(input_value[0] + 0j, device=nn.device)
        if len(input_value) == 2:
            return torch.tensor(input_value[0] + 1j*input_value[1], device=nn.device)
        else:
            raise ValueError("List or tuple input must have maximum length 2.")
    elif torch.is_tensor(input_value):
        if input_value.nelement() == 1:
            return torch.tensor(input_value.item() + 0j, device=nn.device).unsqueeze(0)
        if input_value.nelement() == 2:
            return input_value[0] + 1j*input_value[1]
        else:
            raise ValueError("Tensor input must have maximum 2 elements.")
    else:
        raise ValueError("Input value must be a float, complex, tuple, list or torch.tensor of length 1 or 2.")


def get_complex_function(func):
    """
    | Convert the input function to the unified format (i.e., callable: complex :class:`torch.tensor` -> complex :class:`torch.tensor`).
    | The library allows to define some functions in multiple and flexible ways (callables, constants, lists, tensors). 
      Then, this method includes all the possible definitions and unify the type of the generic function. 

    :param func: Generic input function.
    :type func: int/float/complex/list/tuple/:class:`torch.tensor`/callable
    :returns: 
        - **new_func** (callable) - Copy of the input function in the unified format.
    """
    if callable(func):
        if len(inspect.getfullargspec(func)[0]) == 1:
            output = func(torch.tensor([0.j]))
            if isinstance(output, (int,float,complex)):
                new_func = lambda z: func(z) + 0*z 
            elif torch.is_tensor(output):
                if output.nelement() == 1:
                    new_func = lambda z: func(z) + 0*z
                elif output.nelement() == 2:
                    new_func = lambda z: func(z)[0] + 1.j*func(z)[1] + 0*z
            elif isinstance(output, (tuple,list)):
                    new_func = lambda z: func(z)[0] + 1.j*func(z)[1] + 0*z
            else:
                raise ValueError("No suitable combination found for the output of input function.")      
        elif len(inspect.getfullargspec(func)[0]) == 2:
            output = func(torch.tensor([0.]),torch.tensor([0.]))
            if isinstance(output, (int,float,complex)):
                new_func = lambda z: func(z) + 0*z 
            elif torch.is_tensor(output):
                if output.nelement() == 1:
                    new_func = lambda z: func(z.real,z.imag) + 0*z
                elif output.nelement() == 2:
                    new_func = lambda z: func(z.real,z.imag)[0] + 1.j*func(z.real,z.imag)[1] + 0*z
            elif isinstance(output, (tuple,list)):
                    new_func = lambda z: func(z.real,z.imag)[0] + 1.j*func(z.real,z.imag)[1] + 0*z
            else:
                raise ValueError("No suitable combination found for the output of input function.")      
        else:
            raise ValueError("Input function must accept either 1 complex input or 2 real inputs.")
    elif isinstance(func, (int, float, complex)):
        new_func = lambda z: func + 0*z
    elif isinstance(func, (list, tuple, torch.tensor)):
        func_pt = torch.tensor(func)
        if func_pt.nelement() == 1:
            func_pt = func_pt + 0j
        elif func_pt.nelement() == 2:
            func_pt = func_pt[0] + 1j*func_pt[1]
        else:
            raise ValueError("List/tuple/tensor input cannot have more than 2 dimensions.")
        new_func = lambda z: func_pt + 0*z
    else:
        raise ValueError("Input function must be a callable, a scalar, or a pair of values.")
    return new_func


def MSE(value, true_value=None):
    """
    Mean squared error (MSE). Equivalent to torch.nn.MSELoss() except it takes into account empty inputs.

    :param value: Input to evaluate.
    :type value: :class:`torch.tensor`
    :param true_value: Reference value to compare the input with. If unassigned, it's by default zero.
    :type value: :class:`torch.tensor`
    :returns: **mse** (float) - MSE error.
    """
    if value.nelement() == 0:
        return 0.
    if true_value is None:
        true_value = torch.zeros_like(value)
    return torch.nn.MSELoss(reduction='sum')(value, true_value)


def PIHNNloss(boundary, model, t):
    """
    Evaluation of the loss function as Mean squared error (MSE).

    :param boundary: Domain boundary, needed to extract training and test points.
    :type boundary: :class:`pihnn.geometries.boundary`
    :param model: Neural network model.
    :type model: :class:`pihnn.nn.PIHNN`/:class:`pihnn.nn.DD_PIHNN`
    :param t: Option for 'training' or 'test'.
    :type t: str
    :returns: **loss** (float) - Computed loss.
    """
    if(model.PDE == 'laplace' or model.PDE == 'biharmonic'):
        return scalar_loss(boundary, model, t)
    elif(model.PDE == 'km' or model.PDE == 'km-so'):
        return km_loss(boundary, model, t)
    else:
        raise ValueError("'model.PDE' must be either 'laplace', 'biharmonic', 'km' or 'km-so'.")
 

def scalar_loss(boundary, model, t):
    """
    Called by :func:`pihnn.utils.PIHNNloss` if one aims to solve the Laplace or biharmonic problem. 

    :param boundary: Domain boundary, needed to extract training and test points.
    :type boundary: :class:`pihnn.geometries.boundary`
    :param model: Neural network model.
    :type model: :class:`pihnn.nn.PIHNN`/:class:`pihnn.nn.DD_PIHNN`
    :param t: Option for 'training' or 'test'.
    :type t: str
    :returns: **loss** (float) - Computed loss.
    """
    if isinstance(model, nn.DD_PIHNN):
        z, normals, bc_idxs, bc_values, mask, twins = boundary(t, dd=True)
    else:
        z, normals, bc_idxs, bc_values = boundary(t)

    if model.PDE in ['laplace','biharmonic']:
        u = model(z.requires_grad_(True), real_output=True)
    else:
        raise ValueError("'model.PDE' must be 'laplace' or 'biharmonic' for this type of loss.")

    if isinstance(model, nn.DD_PIHNN):
        u[twins[0],twins[2]] -= u[twins[1],twins[3]]
        u[twins[1],twins[3]] = 0
        u[mask] = 0
    
    L = 0.
    for j,bc_type in enumerate(boundary.bc_types):
        L += bc_type.scaling_coeff*MSE(bc_type(z, u, normals, bc_values)[bc_idxs==j])
    bc.reset_variables()
    return L / z.nelement()


def km_loss(boundary, model, t):
    """
    Called by :func:`pihnn.utils.PIHNNloss` if one aims to solve the linear elasticity problem 
    through the Kolosov-Muskhelishvili representation. 

    :param boundary: Domain boundary, needed to extract training and test points.
    :type boundary: :class:`pihnn.geometries.boundary`
    :param model: Neural network model.
    :type model: :class:`pihnn.nn.PIHNN`/:class:`pihnn.nn.DD_PIHNN`
    :param t: Option for 'training' or 'test'.
    :type t: str
    :returns: **loss** (float) - Computed loss.
    """

    if isinstance(model, nn.DD_PIHNN):
        z, normals, bc_idxs, bc_values, mask, twins = boundary(t, dd=True)
    else:
        z, normals, bc_idxs, bc_values = boundary(t)
    
    vars = model(z.requires_grad_(True), real_output=True)

    if isinstance(model, nn.DD_PIHNN):
        vars[:,twins[0],twins[2]] -= vars[:,twins[1],twins[3]]
        vars[:,twins[1],twins[3]] = 0
        vars[:,mask] = 0

    sig_xx, sig_yy, sig_xy, u_x, u_y = vars

    L = 0.
    for j,bc_type in enumerate(boundary.bc_types):
        L += MSE(bc_type(z, sig_xx, sig_yy, sig_xy, u_x, u_y, normals, bc_values)[bc_idxs==j])
    bc.reset_variables()
    return L / z.nelement()


def rotate_stresses(vars, angle):
    """
    Stresses and displacements transformation for rotated systems of coordinates.

    :param vars: Tensor containing either :math:`\sigma_{xx},\sigma_{yy},\sigma_{xy},u,v` or :math:`\sigma_{xx},\sigma_{yy},\sigma_{xy}`. In the latter case only stresses are rotated.
    :type vars: :class:`torch.tensor`
    :param angle: Angle corresponding to the rotation of the original system of coordinates.
    :type angle: float
    :returns: 
        - **vars_roto** (:class:`torch.tensor`) - Variables evaluated in the rotated system of coordinates.  
    """
    c = np.cos(angle)
    s = np.sin(angle)
    vars_roto = torch.zeros_like(vars)

    vars_roto[0] = vars[0]*c*c + vars[1]*s*s + 2*vars[2]*c*s
    vars_roto[1] = vars[0]*s*s + vars[1]*c*c - 2*vars[2]*c*s
    vars_roto[2] = (vars[1]-vars[0])*c*s + vars[2]*(c*c-s*s)

    if vars.shape[0] == 5: # I.e., also displacements have to be rotated
        vars_roto[3] = -vars[3]*c + vars[4]*s
        vars_roto[4] =  vars[3]*s + vars[4]*c

    return vars_roto


def compute_J_integral(model, tip, radius=1., integration_points=1000, crack_curve=None):
    """
    Compute the J-integral.
    
    .. math::
        J = \oint_{\gamma} \left(w n_x - (\sigma \cdot n)\cdot\\frac{\partial u}{\partial x}\\right) d\gamma,
    
    where :math:`\gamma` is a closed curve around a crack tip and :math:`w=\\frac{1}{2}\sigma:\\varepsilon`.
    In particular, a first-order discretization is used and :math:`\gamma` is taken as a circle around the crack tip plus, if required, the crack faces.

    :param model: Neural network model, the function is meaningful only if model.PDE is either 'km' or 'km-so'.
    :type model: :class:`pihnn.nn.PIHNN`/:class:`pihnn.nn.DD_PIHNN`
    :param tip: Crack tip where the J-integral is evaluated.
    :type tip: :class:`pihnn.geometries.crack_tip`
    :param radius: Radius of the circumference where the J-integral is evaluated.
    :type radius: float
    :param integration_points: Number of integration points on the circumference.
    :type integration_points: int
    :param crack_curve: Curve associated to the crack tip, for the evaluation of the I-integral on the crack faces, if needed.
    :type crack_curve: :class:`pihnn.geometries.curve`
    :returns: 
        - **J** (float) - Evaluation of the J-integral.
    """

    N_c = integration_points
    P = tip.coords
    z = P + radius*torch.exp(1j*torch.linspace(-np.pi+1e-8,torch.pi+1e-8,N_c+1,device=nn.device)[:-1]).requires_grad_(True)
    ds = 2*np.pi*radius/N_c # Differential for integration
    n = (z-tip.coords)/radius # Normal vector

    if crack_curve is not None:
        z1, n1 = crack_curve.map_point(torch.linspace(0.1/N_c,1-0.1/N_c,N_c,device=nn.device))
        n1 = n1[torch.abs(z1-P)<=radius]
        z1 = z1[torch.abs(z1-P)<=radius]
        z2 = z1
        z1 = z1-1e-9*n1
        z2 = z2+1e-9*n1
        z = torch.cat((z,z1,z2)).requires_grad_(True)
        n = torch.cat((n,n1,-n1))
        ds = torch.abs(z[1:]-z[:-1])
        ds[N_c-1] = 0 # Remove possible discontinuities between z,z1,z2
        ds[N_c-1+int(z1.shape[0])] = 0
        z = z[:-1]
        n = n[:-1]

    # Second part: computation of J-integral
    sxx,syy,sxy,_,uy = model(z, real_output=True, force_williams=True)

    C1 = (1-model.material["poisson"]**2)/model.material["young"]
    C2 = -model.material["poisson"]*(1+model.material["poisson"])/model.material["young"]

    exx = C1*sxx + C2*syy
    eyy = C1*syy + C2*sxx
    exy = sxy/2/model.material["mu"]

    ux_x = exx # All derivatives of u can be obtained from the strain tensor except the anti-symmetric part
    uy_y = eyy
    uy_x = 2*torch.real(nn.derivative(uy,z))
    ux_y = 2*exy-uy_x

    w = 0.5*(sxx*exx + syy*eyy + 2*sxy*exy) # Strain energy density
    sn = (sxx*n.real + sxy*n.imag) + 1j*(sxy*n.real + syy*n.imag) # sigma \cdot n

    norm_t = n.real*tip.norm.real + n.imag*tip.norm.imag
    ux_t = ux_x*tip.norm.real + ux_y*tip.norm.imag
    uy_t = uy_x*tip.norm.real + uy_y*tip.norm.imag
    J = torch.sum(w*norm_t*ds) - torch.sum((sn.real*ux_t+sn.imag*uy_t)*ds)
    return J


def yau_wang_method(model, tip, radius=1., integration_points=1000, crack_curve=None):
    """
    Implementation of the method from `Yau et al. [1980] <https://doi.org/10.1115/1.3153665>`_ to evaluate the
    stress intensity factors (SIFs) in mixed mode I-II from the J-integral.

    As explained in :func:`pihnn.utils.compute_J_integral`, the J-integrals are computed with a first-order discretization scheme along a circumference centered in the crack tip.

    :param model: Neural network model, the function is meaningful only if model.PDE is either 'km' or 'km-so'.
    :type model: :class:`pihnn.nn.PIHNN`/:class:`pihnn.nn.DD_PIHNN`
    :param tip: Crack tip where the J-integral is evaluated.
    :type tip: :class:`pihnn.geometries.crack_tip`
    :param radius: Radius of the circumference where the J-integrals are computed.
    :type radius: float
    :param integration_points: Number of integration points on the circumference.
    :type integration_points: int 
    :param crack_curve: Curve associated to the crack tip, for the evaluation of the I-integral on the crack faces, if needed.
    :type crack_curve: :class:`pihnn.geometries.curve`
    :returns: 
        - **K_I** (float) - :math:`K_I` stress intensity factor.
        - **K_II** (float) - :math:`K_{II}` stress intensity factor.
    """
    C1 = (1-model.material["poisson"]**2)/model.material["young"]

    net_sif = model.sif
    if torch.norm(net_sif)>1e-5 and model.enrichment=='rice':
        warnings.warn("Rice-enriched network has non-zero Williams SIFs. Are you sure this is correct?")

    aux_sif = torch.zeros_like(net_sif)
    j = 0
    for crack in model.cracks:
        for t in crack.tips:
            if torch.abs(t.coords-tip.coords)<1e-5:
                idx = j
            j+=1

    # NN J integral
    J = compute_J_integral(model, tip, radius, integration_points, crack_curve)
    J_aux = C1

    # First superposed J integral
    aux_sif[idx] = 1.
    model.sif = torch.nn.Parameter(net_sif + aux_sif) if model.enrichment=="williams" else aux_sif
    J_tot = compute_J_integral(model, tip, radius, integration_points, crack_curve)
    I_I = J_tot-J-J_aux

    #Second superposed J integral
    aux_sif[idx] = 1.j
    model.sif = torch.nn.Parameter(net_sif + aux_sif) if model.enrichment=="williams" else aux_sif
    J_tot = compute_J_integral(model, tip, radius, integration_points, crack_curve)
    I_II = J_tot-J-J_aux

    model.sif = net_sif # Restore initial parameters

    K_I = 0.5 * I_I / C1
    K_II = 0.5 * I_II / C1
    return K_I, K_II


def train(boundary, model, n_epochs, learn_rate=1e-3, scheduler_apply=[], scheduler_gamma=0.5, dir="results/", apply_adaptive_sampling=0):
    """
    Performs the training of the neural network.

    :param boundary: Domain boundary, needed to extract training and test points.
    :type boundary: :class:`pihnn.geometries.boundary`
    :param model: Neural network model.
    :type model: :class:`pihnn.nn.PIHNN`/:class:`pihnn.nn.DD_PIHNN`
    :param n_epochs: Number of total epochs.
    :type n_epochs: int
    :param learn_late: Initial learning rate for the optimizer.
    :type learn_rate: float
    :param scheduler_apply: At which epoch to apply the :class:`torch.optim.lr_scheduler.ExponentialLR` scheduler.
    :type scheduler_apply: list of int
    :param scheduler_gamma: Scheduler exponential rate.
    :type scheduler_gamma: float
    :param dir: Directory where to save outputs.
    :type dir: str
    :param apply_adaptive_sampling: At which epoch to apply the RAD adaptive sampling (:func:`pihnn.utils.RAD_sampling`). Leave it to 0 for no adaptive sampling.
    :type apply_adaptive_sampling: int
    :returns: 
        - **loss_epochs** (list of float) - List containing the training loss at each epoch.
        - **loss_epochs_test** (list of float) - List containing the test loss at each epoch.
     """
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)
    loss_epochs = []
    loss_epochs_test = []

    for bc_type in boundary.bc_types:
        if model.PDE in ['laplace','biharmonic'] and not issubclass(bc_type.__class__,bc.scalar_bc):
            raise ValueError("'laplace' and 'biharmonic' problems require boundary conditions derived from 'scalar_bc'.")
        elif model.PDE in ['km','km-so'] and not issubclass(bc_type.__class__,bc.linear_elasticity_bc):
            raise ValueError("Linear elasticity problems require boundary conditions derived from 'linear_elasticity_bc'.")

    for epoch_id in (pbar:=tqdm(range(n_epochs))):

        if epoch_id == apply_adaptive_sampling and epoch_id != 0:
            RAD_sampling(boundary, model)

        optimizer.zero_grad()
        model.zero_grad()
        loss = PIHNNloss(boundary, model, "training")
        loss.backward(retain_graph=True)
        optimizer.step()
        loss_epochs.append(loss.cpu().data)
        loss_test = PIHNNloss(boundary, model, "test")
        loss_epochs_test.append(loss_test.cpu().data)

        with torch.autograd.no_grad():
            if (epoch_id % 10 == 0):
                pbar.set_postfix_str("training loss: "+"%.2E" % loss_epochs[-1]+", test loss: "+"%.2E" % loss_epochs_test[-1])
            if (epoch_id in scheduler_apply):
                scheduler.step()

    loss = np.column_stack((loss_epochs, loss_epochs_test))
    if dir[-1] != "/":
        dir = dir.append("/")
    if not os.path.exists(dir):
        os.mkdir(dir)
        print("# Created path "+os.path.abspath(dir))
    np.savetxt(dir+"loss.dat", loss)
    print("# Saved loss at "+os.path.abspath(dir+"loss.dat"))
    torch.save(model.state_dict(), dir+"model.dict")
    print("# Saved neural network model at "+os.path.abspath(dir+"model.dict"))

    return loss_epochs, loss_epochs_test


def RAD_sampling(boundary, model, fine_grid=10000):
    """
    Employ the RAD adaptive sampling from `Wu et al. [2023] <https://doi.org/10.1016/j.cma.2022.115671>`_.

    :param boundary: Domain boundary, needed to extract training and test points.
    :type boundary: :class:`pihnn.geometries.boundary`
    :param model: Neural network model.
    :type model: :class:`pihnn.nn.PIHNN`/:class:`pihnn.nn.DD_PIHNN`
    :param fine_grid: Number of points to sample for obtaining the residual statistical information.
    :type fine_grid: int
    """
    z, normals, bc_idxs, bc_values = boundary.extract_points(fine_grid)
    epsilon = torch.zeros_like(z, dtype=torch.float64)
    
    idx = torch.tensor([], dtype=bc_idxs.dtype)

    for j,bc_type in enumerate(boundary.bc_types):
        if(model.PDE == 'laplace' or model.PDE == 'biharmonic'):
            u = model(z[bc_idxs==j].requires_grad_(True), real_output=True)
            epsilon = bc_type(z[bc_idxs==j], u, normals[bc_idxs==j], bc_values[bc_idxs==j])
        else:
            sig_xx, sig_yy, sig_xy, u_x, u_y = model(z[bc_idxs==j].requires_grad_(True), real_output=True)
            epsilon = bc_type(z[bc_idxs==j], sig_xx, sig_yy, sig_xy, u_x, u_y, normals[bc_idxs==j], bc_values[bc_idxs==j])

        epsilon = epsilon/torch.mean(epsilon) + 1
        epsilon /= torch.sum(epsilon)

        num_samples = int(boundary.np_train*torch.sum(bc_idxs==j)/torch.sum(bc_idxs>=0))
        idx_bc = epsilon.multinomial(num_samples=num_samples)
        idx = torch.cat([idx,torch.where(bc_idxs==j)[0][idx_bc]])

    boundary.points_train, boundary.normals_train, boundary.bc_idxs_train, boundary.bc_values_train = z[idx], normals[idx], bc_idxs[idx], bc_values[idx]


def compute_Lp_error(triangulation, model, model_true, p=2):
    """
    Compute and print to screen the approximated relative :math:`L^p` error between a model and a reference solution. I.e.,

    .. math::
        \\frac{\|u_{NN}-u\|_{L^p(\Omega)}}{\|u\|_{L^p(\Omega)}} = \left(\\frac{\int_{\Omega} |u_{NN}-u|^p}{\int_{\Omega} |u|^p} \\right)^{1/p} 
        \\approx \left(\\frac{\sum_{(x,y)\in \mathcal{T}} |u_{NN}(x,y)-u(x,y)|^p}{\sum_{(x,y)\in \mathcal{T}} |u(x,y)|^p} \\right)^{1/p},

    where :math:`u` denotes any of the variables of interest and :math:`\mathcal{T}` is the set of points in the triangulation. 

    :param triangulation: 2D mesh used for evaluating the model.
    :type triangulation: :class:`matplotlib.tri.Triangulation` 
    :param model: Neural network model.
    :type model: :class:`pihnn.nn.PIHNN`/:class:`pihnn.nn.DD_PIHNN`
    :param model_true: Reference solution. It must be a scalar function for Laplace and biharmonic problems whereas it must return the 3 components of the stress tensor when solving the linear elasticity problem.
    :type model_true: callable
    :param p: Exponent for the :math:`L^p` error.
    :type p: int
    :returns: **errors** (list of float) - Approximated :math:`L^p` error for each variable of interest.
     """

    z = torch.tensor(triangulation.x + 1.j*triangulation.y).to(nn.device).requires_grad_(True)

    if isinstance(model, nn.PIHNN):
        vars = model(z, real_output=True).detach().cpu()
    else:
        if len(inspect.getfullargspec(model)[0]) == 1: # Check if it's a scalar complex function
            vars = model(z.detach().cpu())
        elif len(inspect.getfullargspec(model)[0]) == 2: # Check if it's a function with two scalar inputs
            vars = model(z.real.detach().cpu(), z.imag.detach().cpu())
        else:
            raise ValueError("'model' must be a function with a scalar complex input or 2D real inputs.")

    if isinstance(model_true, nn.PIHNN):
        vars_true = model_true(z, real_output=True).detach().cpu()
    else:
        if len(inspect.getfullargspec(model_true)[0]) == 1: # Check if it's a scalar complex function
            vars_true = model_true(z.detach().cpu())
        elif len(inspect.getfullargspec(model_true)[0]) == 2: # Check if it's a function with two scalar inputs
            vars_true = model_true(z.real.detach().cpu(), z.imag.detach().cpu())
        else:
            raise ValueError("'model_true' must be a function with a scalar complex input or 2D real inputs.")

    errors = []    
    if torch.is_tensor(vars):
        errors = torch.pow(torch.sum(torch.pow(vars-vars_true,p)),1/p).item() / torch.pow(torch.sum(torch.pow(vars_true,p)),1/p).item()
    else:
        for i in range(min(len(vars),len(vars_true))):
            errors.append(torch.pow(torch.sum(torch.pow(vars[i]-vars_true[i],p)),1/p).item() / torch.pow(torch.sum(torch.pow(vars_true[i],p)),1/p).item())
    return errors