"""Instances and methods for boundary conditions."""
import torch
import pihnn.utils as utils


def reset_variables():
    """
    For efficiency reasons, some variables are stored so that they are computed only once at every forward pass.
    In particular, Neumann boundary conditions employ first derivatives which are expensive to compute.
    Hence, this method is called to free computed derivatives at the end of the forward pass.
    """
    if 'u_z' in globals():
        del globals()['u_z']
    if 'u_zzc' in globals():
        del globals()['u_zzc']



class scalar_bc():
    """
    Base abstract class for all the boundary conditions employed in scalar problems (Laplace, biharmonic).
    """
    def __call__(self,z,u,normal,rhs):
        """
        Calculate residual of the boundary condition.

        :param z: Coordinates of the points where the BC is evaluated.
        :type z: :class:`torch.tensor`
        :param u: Solution evaluated at the 'z' coordinates.
        :type u: :class:`torch.tensor`
        :param normal: Boundary outward normal vectors at the 'z' coordinates.
        :type normal: :class:`torch.tensor`
        :param rhs: Boundary condition RHS assigned value at the 'z' coordinates.
        :type rhs: :class:`torch.tensor`
        :returns: 
            - **error** (:class:`torch.tensor`) - Residual of the boundary condition at the 'z' coordinates.
        """
        raise NotImplementedError



class linear_elasticity_bc():
    """
    Base abstract class for all the boundary conditions employed in linear elasticity.
    """
    def __call__(self,z,sxx,syy,sxy,ux,uy,normal,rhs):
        """
        Calculate residual of the boundary condition.

        :param z: Coordinates of the points where the BC is evaluated.
        :type z: :class:`torch.tensor`
        :param sxx: :math:`\sigma_{xx}` evaluated at the 'z' coordinates.
        :type sxx: :class:`torch.tensor`
        :param syy: :math:`\sigma_{yy}` evaluated at the 'z' coordinates.
        :type syy: :class:`torch.tensor`
        :param sxy: :math:`\sigma_{xy}` evaluated at the 'z' coordinates.
        :type sxy: :class:`torch.tensor`
        :param ux: :math:`u_{x}` evaluated at the 'z' coordinates.
        :type ux: :class:`torch.tensor`
        :param uy: :math:`u_{y}` evaluated at the 'z' coordinates.
        :type uy: :class:`torch.tensor`
        :param normal: Boundary outward normal vectors at the 'z' coordinates.
        :type normal: :class:`torch.tensor`
        :param rhs: Boundary condition RHS assigned value at the 'z' coordinates.
        :type rhs: :class:`torch.tensor`
        :returns: 
            - **error** (:class:`torch.tensor`) - Residual of the boundary condition at the 'z' coordinates.
        """
        raise NotImplementedError
    


class dirichlet_bc(scalar_bc):
    """
    Dirichlet boundary condition: 
    
    .. math::
        u=u_d,
     
    | where :math:`u` is the solution and :math:`u_d` is the assigned RHS value.
    | To be used for the Laplace problem.
    """
    def __call__(self,z,u,normal,rhs):
        return torch.abs(u - rhs)
        


class neumann_bc(scalar_bc):
    """
    Neumann boundary condition: 
    
    .. math::
        \\nabla u \cdot n= u_n,

    | where :math:`u` is the solution, :math:`n` is the outward normal vector and :math:`u_n` is the assigned RHS value.        
    | To be used for the Laplace problem.
    """
    def __call__(self,z,u,normal,rhs):
        if 'u_z' not in globals():
            global u_z
            u_z = utils.derivative(u,z)
        dudN = 2*u_z.real*normal.real - 2*u_z.imag*normal.imag
        return torch.abs(dudN - rhs)



class dirichlet_neumann_bc(scalar_bc):
    """
    Dirichlet-Neumann boundary condition:

    .. math::
        u=u_d, \\nabla u \cdot n= u_n,

    | where :math:`u` is the solution, :math:`n` is the outward normal vector and :math:`u_d,u_n` are the assigned RHS values.        
    | To be used for the biharmonic problem.
    """
    def __call__(self,z,u,normal,rhs):
        if 'u_z' not in globals():
            global u_z
            u_z = utils.derivative(u,z)
        dudn = 2*u_z.real*normal.real - 2*u_z.imag*normal.imag
        return torch.sqrt((u - rhs.real)*(u - rhs.real) + (dudn - rhs.imag)*(dudn - rhs.imag))
    


class dirichlet_laplace_bc(scalar_bc):
    """
    Dirichlet-Laplace boundary condition:

    .. math::
        u=u_d, \\nabla^2 u = u_l,

    | where :math:`u` is the solution and :math:`u_d,u_l` are the assigned RHS values.        
    | To be used for the biharmonic problem.
    """
    def __call__(self,z,u,normal,rhs):
        if 'u_z' not in globals():
            global u_z
            u_z = utils.derivative(u,z)
        if 'u_zzc' not in globals():
            global u_zzc
            u_zzc = utils.derivative(u_z,z,conjugate=True)
        u_laplacian = 4 * u_zzc
        return torch.sqrt((u - rhs.real)*(u - rhs.real) + (u_laplacian - rhs.imag)*(u_laplacian - rhs.imag))



class stress_bc(linear_elasticity_bc):
    """
    Stress boundary condition:

    .. math::
        \sigma \cdot n = t_0,

    | where :math:`\sigma` is the stress tensor, :math:`n` is the outward normal vector and :math:`t_0` is the assigned RHS value (applied traction).        
    | To be used for the linear elasticity problem.
    """
    def __call__(self,z,sxx,syy,sxy,ux,uy,normal,rhs):
        sigNx = normal.real * sxx + normal.imag * sxy
        sigNy = normal.real * sxy + normal.imag * syy
        return torch.sqrt((sigNx - rhs.real)*(sigNx - rhs.real) + (sigNy - rhs.imag)*(sigNy - rhs.imag))



class displacement_bc(linear_elasticity_bc):
    """
    Displacement boundary condition:

    .. math::
        u = u_0,

    | where :math:`u` is the displacement vector and :math:`u_0` is the assigned RHS value.        
    | To be used for the linear elasticity problem.
    """
    def __call__(self,z,sxx,syy,sxy,ux,uy,normal,rhs):
        return torch.sqrt((ux - rhs.real)*(ux - rhs.real) + (uy - rhs.imag)*(uy - rhs.imag))
    


class symmetry_bc(linear_elasticity_bc):
    """
    Symmetry boundary condition:

    .. math::
        (\sigma \cdot n) \\times n = u \cdot n = 0,

    | where :math:`\sigma` is the stress tensor, :math:`u` is the displacement vector and :math:`n` is the outward normal vector.
    | To be used for the linear elasticity problem.
    """
    def __call__(self,z,sxx,syy,sxy,ux,uy,normal,rhs=0):
        displacement = ux*normal.real + uy*normal.imag
        sigNx = normal.real * sxx + normal.imag * sxy
        sigNy = normal.real * sxy + normal.imag * syy
        shear_stress = normal.imag * sigNx - normal.real * sigNy  
        return torch.sqrt(displacement*displacement + shear_stress*shear_stress)



class normal_displacement_bc(linear_elasticity_bc):
    """
    Normal displacement boundary condition:

    .. math::
        u \cdot n = u_0,

    | where :math:`u` is the displacement vector, :math:`n` is the outward normal vector and :math:`u_0` is the assigned RHS value.
    | To be used for the linear elasticity problem.
    """
    def __call__(self,z,sxx,syy,sxy,ux,uy,normal,rhs=0):
        displacement = ux*normal.real + uy*normal.imag
        return torch.abs(displacement)
    


class interface_bc(scalar_bc, linear_elasticity_bc):
    """
    Interface condition for DD-PIHNNs. 
    
    For scalar problems:

    .. math::
        [u] = [\\nabla u \cdot n] = 0,
    
    where :math:`u` is the solution and :math:`n` is the outward normal vector.

    For linear elasticity:

    .. math::
        [u] = [\sigma \cdot n] = 0,

    where :math:`u` is the displacement vector, :math:`\\sigma` is the stress tensor and :math:`n` is the outward normal vector.
    
    In both cases, :math:`[\cdot]` denotes the discontinuity across 2 domains.
    """
    def __call__(self,*args):
        """
        | Calculate residual of the boundary condition.
        | It employs :func:`call_for_scalar` or :func:`call_for_linear_elasticity` based on the types of arguments.
        """
        if len(args) == 4: # z,u,normal,rhs
            return self.call_for_scalar(*args)
        elif len(args) == 8: # self,z,sxx,syy,sxy,ux,uy,normal,rhs
            return self.call_for_linear_elasticity(*args)
        else:
            raise ValueError("No suitable combination of input found for interface boundary condition.")


    def call_for_scalar(self,z,u,normal,rhs):
        """
        Calculate residual of the boundary condition (for scalar problems).

        :param z: Coordinates of the points where the BC is evaluated.
        :type z: :class:`torch.tensor`
        :param u: Solution evaluated at the 'z' coordinates.
        :type u: :class:`torch.tensor`
        :param normal: Boundary outward normal vectors at the 'z' coordinates.
        :type normal: :class:`torch.tensor`
        :param rhs: Boundary condition RHS assigned value at the 'z' coordinates.
        :type rhs: :class:`torch.tensor`
        :returns: 
            - **error** (:class:`torch.tensor`) - Residual of the boundary condition at the 'z' coordinates.
        """
        if 'u_z' not in globals():
            global u_z
            u_z = utils.derivative(u,z)
        dudN = 2*u_z.real*normal.real - 2*u_z.imag*normal.imag
        return torch.sqrt(u*u + dudN*dudN)


    def call_for_linear_elasticity(self,z,sxx,syy,sxy,ux,uy,normal,rhs):
        """
        Calculate residual of the boundary condition (for linear elasticity problems).

        :param z: Coordinates of the points where the BC is evaluated.
        :type z: :class:`torch.tensor`
        :param sxx: :math:`\sigma_{xx}` evaluated at the 'z' coordinates.
        :type sxx: :class:`torch.tensor`
        :param syy: :math:`\sigma_{yy}` evaluated at the 'z' coordinates.
        :type syy: :class:`torch.tensor`
        :param sxy: :math:`\sigma_{xy}` evaluated at the 'z' coordinates.
        :type sxy: :class:`torch.tensor`
        :param ux: :math:`u_{x}` evaluated at the 'z' coordinates.
        :type ux: :class:`torch.tensor`
        :param uy: :math:`u_{y}` evaluated at the 'z' coordinates.
        :type uy: :class:`torch.tensor`
        :param normal: Boundary outward normal vectors at the 'z' coordinates.
        :type normal: :class:`torch.tensor`
        :param rhs: Boundary condition RHS assigned value at the 'z' coordinates.
        :type rhs: :class:`torch.tensor`
        :returns: 
            - **error** (:class:`torch.tensor`) - Residual of the boundary condition at the 'z' coordinates.
        """
        sigNx = sxx*normal.real + sxy*normal.imag
        sigNy = sxy*normal.real + syy*normal.imag
        return torch.sqrt(sigNx*sigNx + sigNy*sigNy + ux*ux  + uy*uy)