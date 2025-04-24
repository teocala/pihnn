"""Script for results plotting."""

import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import numpy as np
import pihnn.nn as nn
import pihnn.utils as utils
import os, inspect
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.use('Agg') 



def get_triangulation(boundary, n_points_interior=5000, n_points_boundary=1000):
    """
    Create a 2D mesh/triangulation from the definition of geometry. Useful for :func:`pihnn.graphics.plot_sol`.

    :param boundary: Domain boundary, needed to extract boundary points and evaluate if a point is interior.
    :type boundary: :class:`pihnn.geometries.boundary`
    :param n_points_interior: Number of grid points in the interior of the domain.
    :type n_points_interior: int
    :param n_points_exterior: Number of grid points on the boundary of the domain.
    :type n_points_exterior: int 
    :returns: **tria** (:class:`matplotlib.tri.Triangulation`) - Domain triangulation/mesh.
    """
    points_B = boundary.extract_points(n_points_boundary)[0]
    points_I_x = torch.rand(n_points_interior, device=nn.device)*(boundary.square[1]-boundary.square[0]) + boundary.square[0]
    points_I_y = torch.rand(n_points_interior, device=nn.device)*(boundary.square[3]-boundary.square[2]) + boundary.square[2]
    points_I = points_I_x+1j*points_I_y
    points_I = points_I[boundary.is_inside(points_I)==1]
    mesh = torch.cat((points_B, points_I)).cpu()

    tria = mpl.tri.Triangulation(mesh.real, mesh.imag)
    barycenter = mesh[tria.triangles].sum(dim=1).to(nn.device) / 3.
    tria.set_mask((boundary.is_inside(barycenter)==0).cpu())

    return tria


def single_plot(triangulation, z, levels, title=None, cmap='jet', **kwargs):
    """
    Single contour plot. Used by the other functions in this module.

    :param triangulation: 2D mesh used for the contour plot.
    :type triangulation: :class:`matplotlib.tri.Triangulation` 
    :param z: Function to be plotted.
    :type z: :class:`numpy.ndarray`/:class:`torch.tensor`
    :param levels: Levels corresponding to the contour coloring.
    :type levels: :class:`numpy.ndarray`
    :param title: Title of the plot, None if no title is desired.
    :type title: str
    :param cmap: Predefined matplotlib color map.
    :type cmap: str
    """
    plt.tricontourf(triangulation, z, levels, cmap=cmap)

    if isinstance(levels, int):
        z = np.array(z)
        ticks = np.linspace(np.min(z),np.max(z),kwargs.get('nticks',8))
    else:
        levels = np.array(levels)
        ticks = np.linspace(np.min(levels),np.max(levels),kwargs.get('nticks',8))
    plt.colorbar(ticks=ticks, format=kwargs.get('ticks_format','%.4f'))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    axis = kwargs.get('axis', [r"$x$",r"$y$"])
    if (title is not None):
        plt.title(title)
    plt.xlabel(axis[0])
    plt.ylabel(axis[1])
    plt.tight_layout()


def plot_loss(loss_train, loss_test, figsize=None, format="png", dir="results/"):
    """
    Plot of the training and test loss.

    :param loss_train: Training loss at each epoch.
    :type loss_train: list of float
    :param loss_train: Test loss at each epoch.
    :type loss_train: list of float   
    :param figsize: Size of the :class:`matplotlib.pyplot.figure`.
    :type figsize: tuple of float 
    :param format: Format of the save figure.
    :type format: str
    :param dir: Directory where to save the figure.
    :type dir: str
    """
    if (figsize is not None):
        plt.figure(figsize=figsize)
    else:
        plt.figure()
    plt.semilogy(loss_train,'r-', linewidth=0.5, label='Training')
    plt.semilogy(loss_test,'b--', linewidth=0.5, label='Test')
    plt.legend()
    plt.xlabel('Epoch number')
    plt.ylabel('MSE loss')
    plt.tight_layout()
    plt.savefig(dir+"loss."+format)
    plt.close()
    print("# Saved plot of loss at "+os.path.abspath(dir+"loss."+format))


def plot_sol(triangulation, model, model_true=None, format="png", dir="results/", figsize=None, split=False, **kwargs):
    """
    Plot of the solution from the training of the network. This can be graphically compared with a reference solution (either numerical or analytical).
    The function behaves differently for each problem:
    
    * For the Laplace and biharmonic problems, it shows the PDE solutions. 
    * For linear elasticity, it plots the stresses.

    :param triangulation: 2D mesh used for the contour plot.
    :type triangulation: :class:`matplotlib.tri.Triangulation` 
    :param model: Neural network model.
    :type model: :class:`pihnn.nn.PIHNN`/:class:`pihnn.nn.DD_PIHNN`
    :param model_true: Reference solution. It must be a scalar function for Laplace and biharmonic problems whereas it must return the 3 components of the stress tensor when solving the linear elasticity problem. Instead, leave it to None if no comparison is desired.
    :type model_true: callable / None
    :param format: Format of the save figure.
    :type format: str
    :param dir: Directory where to save the figure.
    :type dir: str
    :param figsize: Size of the :class:`matplotlib.pyplot.figure`. If None, no figure is created.
    :type figsize: tuple of float
    :param split: Whether to split the solutions into multiple files or a single picture.
    :type split: bool
    """
    if(model.PDE in ['laplace', 'biharmonic']):
        plot_scalar(triangulation, model, model_true, format, dir, figsize, split, **kwargs)
    elif(model.PDE in ['km', 'km-so']):
        plot_stresses(triangulation, model, model_true, format, dir, figsize, split, **kwargs)


def plot_scalar(triangulation, model, model_true=None, format="png", dir="results/", figsize=None, split=False, **kwargs):
    """
    Specialization of :class:`pihnn.graphics.plot_sol` for the Laplace and biharmonic problems.

    :param triangulation: 2D mesh used for the contour plot.
    :type triangulation: :class:`matplotlib.tri.Triangulation` 
    :param model: Neural network model.
    :type model: :class:`pihnn.nn.PIHNN`/:class:`pihnn.nn.DD_PIHNN`
    :param model_true: Reference solution as a scalar function (leave it to None if no comparison is desired).
    :type model_true: callable/None
    :param format: Format of the save figure.
    :type format: str
    :param dir: Directory where to save the figure.
    :type dir: str
    :param figsize: Size of the :class:`matplotlib.pyplot.figure`. If None, no figure is created.
    :type figsize: tuple of float 
    :param split: Whether to split the solutions into multiple files or a single picture.
    :type split: bool
    """

    z = torch.tensor(triangulation.x + 1.j*triangulation.y).to(nn.device).requires_grad_(True)
    if isinstance(model, nn.PIHNN):
        u = model(z, real_output=True).detach().cpu()
    else:
        if len(inspect.getfullargspec(model)[0]) == 1:
            u = torch.real(model(z.detach().cpu()))
        elif len(inspect.getfullargspec(model)[0]) == 2:
            u = torch.real(model(z.real.detach().cpu(),z.imag.detach().cpu()))
        else:
            raise ValueError("'model' must accept 1 complex input or 2 real inputs.")

    if (model_true is not None):
        if isinstance(model_true, nn.PIHNN):
            u_true = model_true(z, real_output=True).detach().cpu()
        else:
            if len(inspect.getfullargspec(model_true)[0]) == 1:
                u_true = torch.real(model_true(z.detach().cpu()))
            elif len(inspect.getfullargspec(model_true)[0]) == 2: 
                u_true = torch.real(model_true(z.real.detach().cpu(),z.imag.detach().cpu()))
            else:
                raise ValueError("'model_true' must accept 1 complex input or 2 real inputs.")
        levels = np.linspace(torch.min(torch.minimum(u,u_true)),torch.max(torch.maximum(u,u_true)),50)

    dir0 = dir[:len(dir)-len(dir.partition("/")[-1])]
    if not os.path.exists(dir0):
        os.mkdir(dir0)
        print("# Created path "+os.path.abspath(dir0))

    if not split:
        if (figsize is None and model_true is not None):
            figsize = (12,4)
        elif (figsize is None and model_true is None):
            figsize = (4,4)        
        plt.figure(figsize=figsize)
        if (model_true is not None):
            plt.subplot(1, 3, 1)
            single_plot(triangulation, u_true, kwargs.get('u_levels', levels), r'Exact solution', **kwargs)
            plt.subplot(1, 3, 2)
            single_plot(triangulation, u, kwargs.get('u_levels', levels), r'Learned solution', **kwargs)
            plt.subplot(1, 3, 3)
            single_plot(triangulation, torch.abs(u-u_true), kwargs.get('u_err_levels', 50), r'Error', **kwargs)
        else:
            single_plot(triangulation, u, kwargs.get('u_levels', 50), r'Learned solution', **kwargs)
        plt.savefig(dir+"solution."+format)
        plt.close()
        print("# Saved plot of solution at "+os.path.abspath(dir+"solution."+format))

    else:
        if(figsize is None):
                figsize = (4,4)
        if model_true is not None:
            plt.figure(1, figsize=figsize)
            single_plot(triangulation, u_true, kwargs.get('u_levels', levels), **kwargs)
            plt.savefig(dir+"u_exact."+format)
            plt.figure(2, figsize=figsize)
            single_plot(triangulation, u, kwargs.get('u_levels', levels), **kwargs)
            plt.savefig(dir+"u."+format) 
            plt.figure(3, figsize=figsize)
            single_plot(triangulation, torch.abs(u-u_true), kwargs.get('u_err_levels', 50), **kwargs)
            plt.savefig(dir+"u_error."+format)  
            plt.close('all')
        else:
            plt.figure(figsize=figsize)
            single_plot(triangulation, u, kwargs.get('u_levels', 50), **kwargs)
            plt.savefig(dir+"u."+format) 
            plt.close()          
        print("# Saved plots of solution at "+os.path.abspath(dir))


def plot_stresses(triangulation, model, model_true=None, format="png", dir="results/", figsize=None, split=False, **kwargs):
    """
    Specialization of :class:`pihnn.graphics.plot_sol` for the linear elasticity problem.

    :param triangulation: 2D mesh used for the contour plot.
    :type triangulation: :class:`matplotlib.tri.Triangulation` 
    :param model: Neural network model.
    :type model: :class:`pihnn.nn.PIHNN`/:class:`pihnn.nn.DD_PIHNN`
    :param model_true: Reference solution. It returns the 3 components of the stress tensor (leave it to None if no comparison is desired).
    :type model_true: callable
    :param format: Format of the save figure.
    :type format: str
    :param dir: Directory where to save the figure.
    :type dir: str
    :param figsize: Size of the :class:`matplotlib.pyplot.figure`. If None, no figure is created.
    :type figsize: tuple of float 
    :param split: Whether to split the solutions into multiple files or a single picture.
    :type split: bool
    """

    z = torch.tensor(triangulation.x + 1.j*triangulation.y).to(nn.device).requires_grad_(True)
    if isinstance(model, nn.PIHNN):
        vars_NN = model(z, real_output=True).detach().cpu()
    else:
        if len(inspect.getfullargspec(model)[0]) == 1:
            vars_NN = model(z.detach().cpu())
        elif len(inspect.getfullargspec(model)[0]) == 2:
            vars_NN = model(z.real.detach().cpu(), z.imag.detach().cpu())
        else:
            raise ValueError("'model' must accept 1 complex input or 2 real inputs.")
    nv = vars_NN.shape[0]

    if model_true is not None:
        if isinstance(model_true, nn.PIHNN):
            vars_true = model_true(z, real_output=True).detach().cpu()
        else:
            if len(inspect.getfullargspec(model_true)[0]) == 1:
                vars_true = model_true(z.detach().cpu())
            elif len(inspect.getfullargspec(model_true)[0]) == 2:
                vars_true = model_true(z.real.detach().cpu(), z.imag.detach().cpu())
            else:
                raise ValueError("'model' must accept 1 complex input or 2 real inputs.")
        nv = min(nv, vars_true.shape[0])

    if nv not in [3,5]:
        raise ValueError("Models must return either 'sigma_xx,sigma_yy,sigma_xy,u_x,u_y' or 'sigma_xx,sigma_yy,sigma_xy'.")

    dir0 = dir[:len(dir)-len(dir.partition("/")[-1])]
    if not os.path.exists(dir0):
        os.mkdir(dir0)
        print("# Created path "+os.path.abspath(dir0))
    names_txt = ["sxx", "syy", "sxy", "ux", "uy"]
    names_latex = ["$\sigma_{xx}$", "$\sigma_{yy}$", "$\sigma_{xy}$", "$u_x$", "$u_y$"]

    if isinstance(model, nn.enriched_PIHNN) and kwargs.get('apply_crack_bounds', 0):
        for crack in model.cracks:
            for tip in crack.tips:
                C = torch.sqrt(torch.abs(z.detach()-tip.coords)) / torch.max(torch.sqrt(torch.abs(z.detach()-tip.coords)))
                for i in range(3):
                    vars_NN[i] *= C.cpu()
                    if model_true is not None:
                        vars_true[i] *= C.cpu()

    if(not split):
        if(figsize is None):
            if  model_true is not None:
                figsize = (4*nv,4*3)
            else:
                figsize = (20,4)   
        plt.figure(figsize=figsize)
        if model_true is not None:
            for i in range(nv):
                levels = np.linspace(torch.min(torch.minimum(vars_NN[i],vars_true[i])),torch.max(torch.maximum(vars_NN[i],vars_true[i])),50)
                plt.subplot(3,nv,i+1)
                single_plot(triangulation, vars_true[i], kwargs.get(names_txt[i]+'_levels', levels), r'Exact '+names_latex[i], **kwargs)
                plt.subplot(3,nv,i+nv+1)
                single_plot(triangulation, vars_NN[i], kwargs.get(names_txt[i]+'_levels', levels), r'Learned '+names_latex[i], **kwargs)
                plt.subplot(3,nv,i+2*nv+1)
                single_plot(triangulation, torch.abs(vars_NN[i]-vars_true[i]), kwargs.get(names_txt[i]+'_err_levels', 50), r'Error '+names_latex[i], **kwargs)             
        else:
            for i in range(nv):
                plt.subplot(1, nv, i+1)
                single_plot(triangulation, vars_NN[i], kwargs.get(names_txt[i]+'_levels', 50), r'Learned '+names_latex[i], **kwargs)
        plt.savefig(dir+"solution."+format)
        plt.close()
        print("# Saved plot of stresses "+ ("and displacements " if nv==5 else "") +"at "+os.path.abspath(dir+"solution."+format))

    else:
        if(figsize is None):
                figsize = (4,4)
        for i in range(nv):
            if model_true is not None:
                levels = np.linspace(torch.min(torch.minimum(vars_NN[i],vars_true[i])),torch.max(torch.maximum(vars_NN[i],vars_true[i])),50)
                plt.figure(1, figsize=figsize)
                single_plot(triangulation, vars_true[i], kwargs.get(names_txt[i]+'_levels', levels), **kwargs)
                plt.savefig(dir+names_txt[i]+"_exact."+format)
                plt.figure(2, figsize=figsize)
                single_plot(triangulation, vars_NN[i], kwargs.get(names_txt[i]+'_levels', levels), **kwargs)
                plt.savefig(dir+names_txt[i]+"."+format) 
                plt.figure(3, figsize=figsize)
                single_plot(triangulation, torch.abs(vars_NN[i]-vars_true[i]), kwargs.get(names_txt[i]+'_err_levels', 50), **kwargs)
                plt.savefig(dir+names_txt[i]+"_error."+format)  
                plt.close('all')
            else:
                plt.figure(figsize=figsize)
                single_plot(triangulation, vars_NN[i], kwargs.get(names_txt[i]+'_levels', 50), **kwargs)
                plt.savefig(dir+names_txt[i]+"."+format) 
                plt.close()          
        print("# Saved plots of stresses "+ ("and displacements " if nv==5 else "") +"at "+os.path.abspath(dir))


def plot_training_points(boundary, format="png", dir="results/", figsize=(8,8), markersize=4):
    """
    Generates a plot of the current batch of training points with some information regarding the boundary conditions and the domain (the latter, only for DD-PIHNNs).

    :param boundary: Domain boundary, needed to extract training points.
    :type boundary: :class:`pihnn.geometries.boundary`
    :param format: Format of the save figure.
    :type format: str
    :param dir: Directory where to save the figure.
    :type dir: str
    :param figsize: Size of the :class:`matplotlib.pyplot.figure`. If None, no figure is created.
    :type figsize: tuple of float 
    :param markersize: Size of the marker that is used to plot a point.
    :type markersize: float
    """
    points, _, bc_idxs, _ = boundary("training")
    points = points.cpu().detach()
    bc_idxs = bc_idxs.cpu().detach()

    plt.figure(figsize=figsize) 
    for i in range(len(boundary.bc_names)):
        if(points[bc_idxs==i].nelement() != 0):
            plt.plot(points[bc_idxs==i].real, points[bc_idxs==i].imag, "x", markersize=markersize, label=boundary.bc_names[i])
    plt.legend()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.gca().set_aspect('equal', adjustable='box')

    dir0 = dir[:len(dir)-len(dir.partition("/")[-1])]
    if not os.path.exists(dir0):
        os.mkdir(dir0)
        print("# Created path "+os.path.abspath(dir0))
    plt.savefig(dir+"points_bc."+format)
    plt.close()
    print("# Saved plot of training points at "+os.path.abspath(dir+"points_bc."+format))

    if boundary.dd_partition is not None and boundary.n_domains > 1:
        domains = boundary.dd_partition(points).cpu()
        plt.figure(figsize=figsize)
        for i in range(boundary.n_domains):
            points_d = points[(domains[i,:]==1) & (torch.sum(domains, dim=0)==1)]
            plt.plot(points_d.real, points_d.imag, "x", markersize=markersize, label="Domain "+str(i))
            for j in range(i+1,boundary.n_domains):
                points_d = points[(domains[i,:]==1) & (domains[j,:]==1)]
                if points_d.nelement() != 0:
                    plt.plot(points_d.real, points_d.imag, "x", markersize=markersize, label="Interface domains "+str(i)+","+str(j))
        plt.legend()
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y$")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(dir+"points_dd."+format)
        plt.close()
        print("# Saved plot of domain partitioning at "+os.path.abspath(dir+"points_dd."+format))


def plot_dem(model, crack, tip, max_distance, n_points=100, figsize=None, format='png', dir='results/', markersize=2):
    """
    Calculate and plot stress intensity factors (SIFs) from displacement extrapolation method (DEM).

    :param model: Neural network model, the function is meaningful only if model.PDE is either 'km' or 'km-so'.
    :type model: :class:`pihnn.nn.PIHNN`/:class:`pihnn.nn.DD_PIHNN`
    :param crack: Curve corresponding to the crack to analyse.
    :type crack: class:`pihnn.geometries.crack_line`
    :param tip: Crack tip associated to the SIF.
    :type tip: class:`pihnn.geometries.crack_tip`
    :param max_distance: Maximum distance from the crack tip to consider in the DEM regression.
    :type max_distance: float
    :param n_points: Number of points to consider in the DEM regression.
    :type n_points: int
    :param figsize: Size of the :class:`matplotlib.pyplot.figure`.
    :type figsize: tuple of float 
    :param format: Format of the saved figure.
    :type format: str
    :param dir: Directory where to save the figure.
    :type dir: str
    :param markersize: Size of the marker that is used to plot a point.
    :type markersize: float

    :returns: 
        - **sif_I_ext** (float) - Evaluation of the extrapolated :math:`K_I` stress intensity factor.
        - **sif_II_ext** (float) - Evaluation of the extrapolated :math:`K_{II}` stress intensity factor.
    """
    l = torch.linspace(0, 1, n_points)
    P = tip.P
    z,_ = crack.map_point(l)
    idx = torch.where(torch.abs(z-P)<max_distance)[0]
    if torch.abs(z[0]-P) < 1e-5: # The crack tip coincides with the beginning of the crack line
        a1 = 0
        a2 = idx[-1]/n_points
        z,n = crack.map_point(torch.linspace(a1,a2,n_points)) # First point is next to tip, last point is at max_distance
    else: # The crack tip coincides with the end of the crack line
        a1 = 1
        a2 = idx[0]/n_points
        z,n = crack.map_point(torch.linspace(a1,a2,n_points)) # First point is next to tip, last point is at max_distance
    z = z[1:] # Remove first point to avoid NaNs
    n = n[1:]
    l = l[1:]
    N = n_points-1
    zz = torch.empty([2*N])*1.j
    nn = torch.cat((n,n))
    zz[:N] = z + 1e-6*n
    zz[N:] = z - 1e-6*n
    _,_,_,ux,uy = model(zz.requires_grad_(True), real_output=True)
    un = ux*nn.real + uy*nn.imag
    up = ux*nn.imag - uy*nn.real

    sif_I = (un[:N]-un[N:]).detach() * torch.sqrt(2*torch.pi/torch.abs(P-z))*model.material['mu']/(model.material['km_gamma']+1)
    sif_II = (up[:N]-up[N:]).detach() * torch.sqrt(2*torch.pi/torch.abs(P-z))*model.material['mu']/(model.material['km_gamma']+1)
    slope_I = torch.median((sif_I-sif_I[-1])/l) # Average slope from linear regression
    slope_II = torch.median((sif_II-sif_II[-1])/l)
    sif_I_ext = slope_I + sif_I[-1] # Extrapolated SIF
    sif_II_ext = slope_II + sif_II[-1]

    if (figsize is not None):
        plt.figure(figsize=figsize)
    else:
        plt.figure()
    plt.plot(max_distance*l, sif_I, 'r*', label=r"$K_I$", markersize=markersize)
    plt.plot(max_distance*l, slope_I*(1-l) + sif_I[-1], 'b-', label=r"$K_I$ from DEM")
    plt.legend()
    plt.xlabel('Distance from crack tip')
    plt.ylabel(r'$K_I$')
    plt.tight_layout()
    plt.savefig(dir+"dem_i."+format)
    plt.close()

    if (figsize is not None):
        plt.figure(figsize=figsize)
    else:
        plt.figure()
    plt.plot(max_distance*l, sif_II, 'r*', label=r"$K_{II}$", markersize=markersize)
    plt.plot(max_distance*l, slope_II*(1-l) + sif_II[-1], 'b-', label=r"$K_{II}$ from DEM")
    plt.legend()
    plt.xlabel('Distance from crack tip')
    plt.ylabel(r'$K_{II}$')
    plt.tight_layout()
    plt.savefig(dir+"dem_ii."+format)
    plt.close()

    print("# Saved plots of DEM at "+os.path.abspath(dir))
    return sif_I_ext, sif_II_ext