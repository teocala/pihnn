"""Utilities for building and training PIHNNs."""
import torch
import pihnn.nn as nn
import pihnn.bc as bc
import numpy as np
from tqdm import tqdm
import os, inspect


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
        return torch.tensor(input_value +0j)
    elif isinstance(input_value, (list,tuple)):
        if len(input_value) == 1:
            return torch.tensor(input_value[0] + 0j)
        if len(input_value) == 2:
            return torch.tensor(input_value[0] + 1j*input_value[1])
        else:
            raise ValueError("List or tuple input must have maximum length 2.")
    elif torch.is_tensor(input_value):
        if input_value.nelement() == 1:
            return input_value[0] + 0j
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


def derivative(f, z, holom=False, conjugate=False):
    """
    Compute the derivative :math:`\\frac{df}{dz}` through PyTorch automatic differentiation.
    The method requires that 'f' is obtained from 'z' and that 'z.requires_grad' was set to True.
    If 'z' is complex-valued, the Wirtinger derivative is computed instead:
    :math:`\\frac{\partial f}{\partial z}:= \\frac{1}{2}\left(\\frac{\partial f}{\partial x} - i\\frac{\partial f}{\partial x}\\right)`.

    :param f: Function to derivatate.
    :type value: :class:`torch.tensor`
    :param z: Variable against which to derive.
    :type value: :class:`torch.tensor`
    :param holom: If True, the complex derivative is computed by assuming 'f' to be holomorphic (leading to faster calculation). Meaningful only if 'z' is complex.
    :type holom: bool
    :param conjugate: If True, the second Wirtinger derivative :math:`\\frac{\partial f}{\partial \overline{z}}` is computed instead. Meaningful only if 'z' is complex.
    :returns: **derivative** (:class:`torch.tensor`) - Derivative of 'f' with respect to 'z'.
    """
    if torch.is_floating_point(f):
        f = f + 0.j 
    if torch.is_floating_point(z):
            return torch.autograd.grad(torch.real(f), z, grad_outputs=torch.ones_like(z), create_graph = True)[0] + \
                1j*torch.autograd.grad(torch.imag(f), z, grad_outputs=torch.ones_like(z), create_graph = True)[0]
    elif torch.is_complex(z):
        if(holom): # df/dz = 2d(Re(f))/dz when f is holomorphic => faster calculation
            dfdz = torch.autograd.grad(torch.real(f), z, grad_outputs=torch.ones_like(z.real), create_graph = True)[0]
            if not conjugate:
                dfdz = torch.conj(dfdz) # For some reason, the torch derivative is with respect to z conj
        else:
            dudz = torch.autograd.grad(torch.real(f), z, grad_outputs=torch.ones_like(z.real), create_graph = True)[0]
            dvdz = torch.autograd.grad(torch.imag(f), z, grad_outputs=torch.ones_like(z.real), create_graph = True)[0]
            if not conjugate:
                dudz = torch.conj(dudz)
                dvdz = torch.conj(dvdz)
            dfdz = 0.5*dudz + 0.5j*dvdz # For some reason, the torch derivative is twice the right value
        return dfdz


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
        L += MSE(bc_type(z, u, normals, bc_values)[bc_idxs==j])
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


def train(boundary, model, n_epochs, learn_rate=1e-3, scheduler_apply=[], scheduler_gamma=0.5, dir="results/"):
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
        optimizer.zero_grad()
        model.zero_grad()
        loss = PIHNNloss(boundary, model, "training")
        loss.backward()
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
    dir0 = dir[:len(dir)-len(dir.partition("/")[-1])]
    if not os.path.exists(dir0):
        os.mkdir(dir0)
        print("# Created path "+os.path.abspath(dir0))
    np.savetxt(dir+"loss.dat", loss)
    print("# Saved loss at "+os.path.abspath(dir+"loss.dat"))
    torch.save(model.state_dict(), dir+"model.dict")
    print("# Saved neural network model at "+os.path.abspath(dir+"model.dict"))

    return loss_epochs, loss_epochs_test


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
    for i in range(min(len(vars),len(vars_true))):
        errors.append(torch.pow(torch.sum(torch.pow(vars[i]-vars_true[i],p)),1/p).item() / torch.pow(torch.sum(torch.pow(vars_true[i],p)),1/p).item())
    print("L"+str(p)+" errors:", *errors)
    return errors