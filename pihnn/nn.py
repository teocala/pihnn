"""Definition of PIHNN networks."""

import torch
import math, warnings
import pihnn.utils as utils

torch.set_default_dtype(torch.float64) # This also implies default complex128. It applies to all scripts of the library
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Shared with other scripts of the library

def custom_format_warning(message, category, filename, lineno, line=None): # Customize output of warnings
    """
    Custom formatting for warnings.
    """
    return f"{filename}:{lineno}: {category.__name__}: {message}\n"
warnings.formatwarning = custom_format_warning



def derivative(f, z, holom=False, conjugate=False):
    """
    Compute the derivative :math:`\\frac{df}{dz}` through PyTorch automatic differentiation.
    The method requires that 'f' is obtained from 'z' and that 'z.requires_grad' was set to True.
    If 'z' is complex-valued, the Wirtinger derivative is computed instead:
    :math:`\\frac{\partial f}{\partial z}:= \\frac{1}{2}\left(\\frac{\partial f}{\partial x} - i\\frac{\partial f}{\partial x}\\right)`.

    :param f: Function to derivatate.
    :type f: :class:`torch.tensor`
    :param z: Variable against which to derive.
    :type z: :class:`torch.tensor`
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



class ComplexLinear(torch.nn.Module):
    """
    Extension of :class:`torch.nn.Linear` to complex values. 
    
    So, a complex linear layer performs the operation
    
    .. math::
        y = Wx + b,

    where the input :math:`x` is a complex vector of dimension in_features, the output :math:`y` and the bias :math:`b` 
    are complex vectors of dimension out_features, and the complex-valued weight tensor :math:`W` has dimensions (in_features,out_features).
    """
    def __init__(self, stack_features, in_features, out_features, n_domains=None, has_bias=True):
        """
        :param in_features: Number of input units at the current layer.
        :type in_features: int
        :param out_features: Number of output units at the current layer.
        :type out_features: int
        :param has_bias: True if the current layer includes the bias term.
        :type has_bias: bool
        """
        super(ComplexLinear, self).__init__()
        self.is_dd = (n_domains is not None)
        self.has_bias = has_bias
        self.in_features = in_features
        if(self.is_dd):
            self.W = torch.nn.Parameter(torch.empty((stack_features, n_domains, out_features, in_features), dtype=torch.complex128, device=device))
        else:
            self.W = torch.nn.Parameter(torch.empty((stack_features, out_features, in_features), dtype=torch.complex128, device=device))
        if(self.has_bias):
            if(self.is_dd):
                self.B = torch.nn.Parameter(torch.empty((stack_features, n_domains, out_features, 1), dtype=torch.complex128, device=device))
            else:
                self.B = torch.nn.Parameter(torch.empty((stack_features, out_features, 1), dtype=torch.complex128, device=device))


    def init(self, scaling):
        """
        Re-initialization of weights and bias.

        Initialization is defined as a scaled complex-valued He initialization (`Trabelsi et al. [2018] <https://arxiv.org/abs/1705.09792>`_):

        .. math::
            \\text{Re}(w),\\text{Im}(w) &\sim \mathcal{N}\left(0,\\frac{\\texttt{scaling}}{2 \\texttt{in_features}}\\right), \\\\
            bias &=0.

        This allows us to easily include the initialization strategy from `Calafà et al. [2024] <https://doi.org/10.1016/j.cma.2024.117406>`_, Section 3.2.4.
        
        :param scaling: Scaling in the He initialization.
        :type scaling: float
        """
        torch.nn.init.normal_(self.W, 0., math.sqrt(scaling)/math.sqrt(2*self.in_features))
        if (self.has_bias):
            torch.nn.init.constant_(self.B, 0.)


    def weight(self):
        """
        :returns: **weight** (:class:`torch.tensor`) - Tensor with weights.
        """
        return self.W


    def bias(self):
        """
        :returns: **bias** (:class:`torch.tensor`) - Bias vector.
        """
        return self.B


    def forward(self, input):
        """
        Forward step :math:`y=Wx+b`.

        :param input: The input vector :math:`x`.
        :type input: :class:`torch.tensor`
        :returns: **output** (:class:`torch.tensor`) - The output vector :math:`y`.
        """
        if len(input.shape)==1 and not self.is_dd and self.W.shape[2]==1: # First layer in PIHNN networks
            if(self.has_bias):
                return torch.einsum('abc,d->abd', self.W, input) + self.B
            else:
                return torch.einsum('abc,d->abd', self.W, input)     
        elif len(input.shape)==2 and self.is_dd and self.W.shape[3]==1: # First layer in DD_PIHNN network
            if(self.has_bias):
                return torch.einsum('abcd,be->abce', self.W, input) + self.B
            else:
                return torch.einsum('abcd,be->abce', self.W, input)    
        elif len(input.shape)==3 and not self.is_dd: # All layers in PIHNN networks
                if(self.has_bias):
                    return torch.einsum('abc,acd->abd', self.W, input) + self.B
                else:
                    return torch.einsum('abc,acd->abd', self.W, input)    
        elif len(input.shape)==4 and self.is_dd: # All layers in DD_PIHNN networks
            if(self.has_bias):
                return torch.einsum('abcd,abde->abce', self.W, input) + self.B
            else:
                return torch.einsum('abcd,abde->abce', self.W, input)   
        else:
            raise ValueError("Sizes of input and weight tensors do not coincide.")



class ComplexParameter(torch.nn.Module):
    """
    Extension of :class:`torch.nn.Parameter` to complex values. 
    """
    def __init__(self, minvalue=-1-1j, maxvalue=1+1j, in_features=1, out_features=1):
        """
        :param minvalue: The minimum value in the parameter initialization. Specifically, parameters are initialized to :math:`p \sim \mathcal{U}(-\\texttt{minvalue},\\texttt{maxvalue})`.
        :type minvalue: complex
        :param maxvalue: The minimum value in the parameter initialization. See above.
        :type maxvalue: complex
        :param in_features: First size of parameter tensor.
        :type in_features: int
        :param out_features: Second size of parameter tensor.
        :type out_features: int
        """
        super(ComplexParameter, self).__init__()
        self.p_r = torch.nn.Parameter((maxvalue.real-minvalue.real) * torch.rand(in_features, out_features) + minvalue.real)
        self.p_i = torch.nn.Parameter((maxvalue.imag-minvalue.imag) * torch.rand(in_features, out_features) + minvalue.imag)


    def forward(self):
        """
        It returns the parameter tensor.

        :returns: **parameter** (:class:`torch.tensor`) - The tensor of size (in_features,out_features) with the parameters.
        """
        return self.p_r + 1.j * self.p_i



class PIHNN(torch.nn.Module):
    """
    Main class for the employment of physics-informed holomorphic neural networks (PIHNNs) from `Calafà et al. [2024] <https://doi.org/10.1016/j.cma.2024.117406>`_.

    PIHNNs are able to solve 4 types of problems, where :math:`\\varphi,\psi` denote the holomorphic output(s) of the network:

    * 2D Laplace problem ('laplace'): 
        .. math::
            \\nabla^2u=0 \\Leftrightarrow u=\\text{Re}(\\varphi).
    * 2D biharmonic problem with Goursat representation ('biharmonic'): 
        .. math::
            \\nabla^4u=0 \\Leftrightarrow u=\\text{Re}((x-iy)\\varphi + \psi).
    * 2D linear elasticity with Kolosov-Muskhelishvili representation ('km'):
        :math:`\sigma_{xx},\sigma_{yy},\sigma_{xy},u_x,u_y` solve the 2D linear elasticity problem :math:`\\Leftrightarrow`

        .. math::
            \\begin{cases}
            \sigma_{xx} + \sigma_{yy} = 4 \\text{Re}(\\varphi'),  \\\\
            \sigma_{yy} - \sigma_{xx} + 2i\sigma_{xy} = (\overline{z}\\varphi''+\psi'),  \\\\
            2\mu(u_x + iu_y) = \gamma \\varphi - z \overline{\\varphi'} - \overline{\psi},
            \end{cases} \\\\

        where :math:`\\mu` is the shear modulus and :math:`\gamma` is the Kolosov constant.
    * 2D linear elasticity with Kolosov-Muskhelishvili representation, stress-only ('km-so'):
        :math:`\sigma_{xx},\sigma_{yy},\sigma_{xy}` solve the 2D linear elasticity problem :math:`\\Leftrightarrow`

        .. math::
            \\begin{cases}
            \sigma_{xx} + \sigma_{yy} = 4 \\text{Re}(\\varphi),  \\\\
            \sigma_{yy} - \sigma_{xx} + 2i\sigma_{xy} = (\overline{z}\\varphi'+\psi).
            \end{cases} \\\\
    
    The output of the network is therefore the scalar function :math:`\\varphi_{NN}\\approx\\varphi` in the Laplace problem. 
    Instead, for the other problems the PIHNN is composed by 2 stacked networks :math:`\\varphi_{NN},\psi_{NN}\\approx\\varphi,\psi`.
    """
    def __init__(self, PDE, units, material={"lambda": 1, "mu": 1}, activation=torch.exp, has_bias=True, rhs_solution=None):
        """
        :param PDE: Problem to solve, either 'laplace', 'biharmonic', 'km' or 'km-so'.
        :type PDE: str
        :param units: List containing number of units at each layer, e.g., [1,10,10,1].
        :type units: list of int
        :param material: Properties of the material, dictionary with 'lambda' (first Lamé coefficient), 'mu' (second Lamé coefficient).
        :type material: dict
        :param activation: Activation function, by default the complex exponential.
        :type activation: callable
        :param has_bias: True if the linear layers include bias vectors.
        :type has_bias: bool
        :param rhs_solution: Particular solution to the non-homogeneous problem. E.g., :math:`x^2+y^2` for :math:`\\nabla^2u=4`.
        :type rhs_solution: callable
        """
        super(PIHNN, self).__init__()

        print("Device: ", device)
        self.PDE = PDE
        if (PDE == 'laplace'):
            self.n_outputs = 1
        elif (PDE in ['biharmonic', 'km', 'km-so']):
            self.n_outputs = 2
        else:
            raise ValueError("'PDE' must be either 'laplace', 'biharmonic', 'km' or 'km-so'.")
        self.n_layers = len(units) - 1
        if (PDE in ['km', 'km-so']):
            self.material = material
            self.material["poisson"] = material["lambda"]/(2*(material["lambda"]+material["mu"]))
            self.material["young"] = material["mu"]*(3*material["lambda"]+2*material["mu"])/(material["lambda"]+material["mu"])
            self.material["bulk"] = (3*material["lambda"]+2*material["mu"])/3
            self.material["km_gamma"] = (material["lambda"]+3*material["mu"])/(material["lambda"] + material["mu"])
            self.material["km_eta"] = -(1-2*self.material["poisson"])/(2*(1-self.material["poisson"]))
            self.material["km_theta"] = -1/(1-2*self.material["poisson"])  
        self.activation = activation
        self.rhs_solution = lambda z: torch.real(utils.get_complex_function(rhs_solution)(z) if rhs_solution is not None else 0*z.real)
        self.layers = torch.nn.ModuleList()
        if (PDE == 'km-so'):
            warnings.warn("You are using the 'stress-only' configuration. Please ensure that boundary conditions are only of stress type.")

        for i in range(self.n_layers):
            self.layers.append(ComplexLinear(self.n_outputs, units[i], units[i+1], has_bias=has_bias))


    def forward(self, z, real_output=False):
        """
        Forward step, i.e., compute:

        .. math::
            \mathcal{L}_{L,t} \circ \phi \circ \mathcal{L}_{L-1,t} \circ \phi \dots \circ \mathcal{L}_{1,t} (z)

        where :math:`z` is the input, :math:`\phi` the activation function and :math:`\{\mathcal{L}_{m,t}\}` the complex linear layers (:class:`pihnn.nn.ComplexLinear`) for each layer :math:`l=1,\dots,L` and stacked network :math:`t=1,\dots,T`.

        :param z: Input of the network, typically a batch of coordinates from the domain boundary.
        :type z: :class:`torch.tensor`
        :param real_output: Whether to provide the output in the real-valued representation.
        :type real_output: bool
        :returns: **phi** (:class:`torch.tensor`) - Output of the network.
        """
        phi = self.layers[0](z)
        for i in range(1, self.n_layers):
            phi = self.layers[i](self.activation(phi))

        if real_output:
            return self.apply_real_transformation(z, phi.squeeze(1)) + self.rhs_solution(z)
        else:
            return phi.squeeze(1)


    def initialize_weights(self, method, beta=0.5, sample=None, gauss=None):
        """
        Initialization of PIHNNs. Implemented methods:

        * Complex-valued He initialization (`Trabelsi et al. [2018] <https://arxiv.org/abs/1705.09792>`_):
            :math:`\\text{Re}(w),\\text{Im}(w)\sim \mathcal{N}\left(0,\\frac{1}{2 \\texttt{in_features}}\\right), \hspace{3mm} bias=0`.
        * Scaled complex-valued He initialization:
            :math:`\\text{Re}(w),\\text{Im}(w)\sim \mathcal{N}\left(0,\\frac{\\texttt{scaling}}{2 \\texttt{in_features}}\\right), \hspace{3mm} bias=0`.
        * PIHNNs ad-hoc initialization with exponential activations:
            See `Calafà et al. [2024] <https://doi.org/10.1016/j.cma.2024.117406>`_, Section 3.2.4.

        :param method: Either 'he', 'he_scaled', 'exp', see description above.
        :type method: str
        :param beta: Scaling coefficient in the scaled He initialization, :math:`\\beta` coefficient in the Calafà initialization, not used in He initialization.
        :type beta: float
        :param sample: Initial sample :math:`x_0` in the Calafà initialization, not used in the other methods.
        :type sample: :class:`torch.tensor`
        :param gauss: :math:`M_e` coefficient in the Calafà initialization, not used in the other methods.
        :type gauss: int
        """
        if (method=='exp'):
            if(gauss==None):
                gauss = self.n_layers
            for i in range(self.n_layers):
                if (i<gauss):
                    scaling = beta/torch.mean(torch.pow(torch.abs(sample),2)).detach()
                else:
                    scaling = beta/math.exp(beta)
                self.layers[i].init(scaling)
                y = self.layers[i](sample)
                sample = self.activation(y)
            return

        if(method=='he'):
            scaling = 1
        elif(method=='he_scaled'):
            scaling = beta
        else:
            raise ValueError("'method' must be either 'exp','he' or 'he_scaled'.")
        for i in range(self.n_layers):
            self.layers[i].init(scaling)


    def apply_real_transformation(self, z, phi):
        """
        Based on the type of PDE, this method returns the real-valued output from the holomorphic potentials.
        We address to the documentation of :class:`pihnn.nn.PIHNN` for the review of the 4 types of problems and their associated representation.
        
        For PDE = 'laplace' and 'biharmonic', :math:`u` is evaluated at :math:`z`. 
        For PDE = 'km' and 'km-so',  :math:`\sigma_{xx},\sigma_{yy},\sigma_{xy},u_x,u_y` are stacked in a single tensor.
        Finally, in 'km-so', :math:`u_x,u_y` are identically zero.

        :params z: Input of the model, typically a batch of coordinates from the domain boundary.
        :type z: :class:`torch.tensor`
        :param phi: Complex-valued output of the network.
        :type phi: :class:`torch.tensor`
        :returns: **vars** (:class:`torch.tensor`) - Tensor containing the real-valued variable(s) evaluated at :math:`z`. 
        """
        match self.PDE:
            case 'laplace':
                return torch.real(phi)[0]
            
            case 'biharmonic':
                return torch.real(torch.conj(z)*phi[0] + phi[1]) # Goursat representation of biharmonic functions
            
            case 'km' | 'km-so':
                if (self.PDE=='km'): # Normal configuration
                    if phi.shape[0] == 2:
                        phi, psi = phi # The original "phi" actually includes both potentials
                        phi_z = derivative(phi,z,holom=True)
                    elif phi.shape[0] == 3: # When Rice transformation is used
                        phi, psi, phi_z = phi
                    psi_z = derivative(psi,z,holom=True)
                    phi_zz = derivative(phi_z,z,holom=True)
                    tmp = self.material["km_gamma"] * phi - z * torch.conj(phi_z) - torch.conj(psi)
                    u_x = torch.real(tmp) / (2*self.material["mu"])
                    u_y = torch.imag(tmp) / (2*self.material["mu"])
                elif (self.PDE=='km-so'): # Stress-only configuration
                    if phi.shape[0] == 2:
                        phi_z, psi_z = phi # The original "phi" actually includes both potentials
                        phi_zz = derivative(phi_z,z,holom=True)
                    elif phi.shape[0] == 3: # When Rice transformation is used
                        phi_z, psi_z, phi_zz = phi
                    u_x = 0.*torch.abs(phi_z)
                    u_y = 0.*torch.abs(psi_z)
                tmp1 = 2*torch.real(phi_z)
                tmp2 = torch.conj(z)*phi_zz + psi_z
                sig_xx = tmp1 - torch.real(tmp2)
                sig_yy = tmp1 + torch.real(tmp2)
                sig_xy = torch.imag(tmp2)
                return torch.stack([sig_xx,sig_yy,sig_xy,u_x,u_y],0)
            
            case _:
                raise ValueError("model.PDE must be either 'laplace', 'biharmonic', 'km', 'km-so'.")
            


class DD_PIHNN(PIHNN):
    """
    Domain-decomposition physics-informed holomorphic neural networks (DD-PIHNNs).

    DD-PIHNNs have been introduced in `Calafà et al. [2024] <https://doi.org/10.1016/j.cma.2024.117406>`_, Section 4.3, to solve problems on multiply-connected domains.
    The structure is similar to :class:`pihnn.nn.PIHNN` but includes multiple stacked networks, each one corresponding to each function :math:`\\varphi,\psi` and each domain.
    """
    def __init__(self, PDE, units, boundary, material={"lambda": 1, "mu": 1}, activation=torch.exp, has_bias=True, rhs_solution=None):
        """
        :param PDE: Problem to solve, either 'laplace', 'biharmonic', 'km' or 'km-so'.
        :type PDE: str
        :param units: List containing number of units at each layer, e.g., [1,10,10,1].
        :type units: list of int
        :param material: Properties of the material, dictionary with 'lambda' (first Lamé coefficient), 'mu' (second Lamé coefficient).
        :param boundary: Geometry of the domain. Needed for information regarding DD partition.
        :type boundary: :class:`pihnn.geometries.boundary`
        :type material: dict
        :param activation: Activation function, by default the complex exponential.
        :type activation: callable
        :param has_bias: True if the linear layers include bias vectors.
        :type has_bias: bool
        :param rhs_solution: Particular solution to the non-homogeneous problem. E.g., :math:`x^2+y^2` for :math:`\\nabla^2u=4`.
        :type rhs_solution: callable
        """
        super(DD_PIHNN, self).__init__(PDE, units, material, activation, has_bias, rhs_solution)
        if boundary.dd_partition is None:
            raise ValueError("Boundary must be initialized with well-defined 'dd_partition' in order to create DD-PIHNNs.")
        self.dd_partition = boundary.dd_partition
        self.n_domains = boundary.n_domains
        self.layers = torch.nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(ComplexLinear(self.n_outputs, units[i], units[i+1], self.n_domains, has_bias=has_bias))
        if (PDE == 'km-so') and self.n_domains > 1:
            raise ValueError("'stress-only' configuration cannot be applied to DD-PIHNNs.")


    def unflatten(self, z_flat, domains):
        """
        Internal operation to transform a 1D batch of coordinates of dimension :math:`N` to the 2D matrix of dimension :math:`[D,N_D]`, 
        where :math:`D` is the number of subdomains and :math:`N_D` is approximately the number of points per subdomain.
        This operation is needed to increase the efficiency of the network. See :func:`pihnn.geometries.boundary.extract_points_dd` for further details.

        :param z_flat: 1D input vector.
        :type z_flat: :class:`torch.tensor`
        :param domains: A 2D tensor such that :math:`T_{i,j}=1` if and only if the :math:`j`-th point belongs to the :math:`i`-th domain, :math:`0` otherwise.
        :type domains: :class:`torch.tensor`
        :returns: **z** (:class:`torch.tensor`) - 2D output vector.
        """
        max_size = torch.max(domains.sum(1))
        z = torch.empty(z_flat.shape[:-1]+(self.n_domains, max_size), dtype=z_flat.dtype, device=device)
        for d in range(self.n_domains): # We do similarly to boundary.extract_points_dd
            z[...,d,:domains[d,:].sum()] = z_flat[...,domains[d,:]]
        return z
    

    def flatten(self, z, domains):
        """
        Inverse operation of :func:`nn.pihnn.DD_PIHNN.unflatten`.

        :param z: 2D input vector.
        :type z: :class:`torch.tensor`
        :param domains: A 2D tensor such that :math:`T_{i,j}=1` if and only if the :math:`j`-th point belongs to the :math:`i`-th domain, :math:`0` otherwise.
        :type domains: :class:`torch.tensor`
        :returns: **z_flat** (:class:`torch.tensor`) - 1D output vector.    
        """
        z_flat = torch.empty(z.shape[:-2]+domains.shape[1:], dtype=z.dtype, device=device)
        for d in range(self.n_domains): # We do the inverse of the previous operation
            z_flat[...,domains[d,:]] = z[...,d,:domains[d,:].sum()]
        return z_flat


    def forward(self, z, flat_output=True, real_output=False):
        """
        Forward step, i.e., compute:

        .. math::
            \mathcal{L}_{L,t,d} \circ \phi \circ \mathcal{L}_{L-1,t,d} \circ \phi \dots \circ \mathcal{L}_{1,t,d} (z)

        where :math:`z` is the input, :math:`\phi` the activation function, :math:`d=1,\dots,D` the domain to which :math:`z` belongs and :math:`\{\mathcal{L}_{l,t,d}\}` the complex linear layers (:class:`pihnn.nn.ComplexLinear`) for each layer :math:`l=1,\dots,L` and stacked network :math:`(t,d)`.

        :param z: Input of the network, typically a batch of coordinates from the domain boundary.
        :type z: :class:`torch.tensor`
        :param flat_output: If True, the output of the network is a 1D/flat vector. Otherwise, the output is a 2D tensor where the first dimension is the number of domains and the second dimension
            is the number of points per domain. The second option is necessary for the training of the network while one can simply consider a flat output in other circumstances. 
            Notice that the output is flat only if the input is also flat.
        :type flat_output: bool
        :param real_output: Whether to provide the output in the real-valued representation.
        :type real_output: bool
        :returns: **phi** (:class:`torch.tensor`) - Output of the network.
        """

        if (len(z.shape)==1):
            domains = self.dd_partition(z)
            z_dd = self.unflatten(z, domains)
        else:
            z_dd = z

        phi = self.layers[0](z_dd)
        for i in range(1, self.n_layers):
            phi = self.layers[i](self.activation(phi))

        phi = phi.squeeze(2)

        if (flat_output and 'domains' in locals()):
            phi = self.flatten(phi, domains)

        if real_output:
            return self.apply_real_transformation(z, phi) + self.rhs_solution(z)
        else:
            return phi


    def initialize_weights(self, method, beta=0.5, sample=None, gauss=None):
        """
        Equivalent to :func:`pihnn.nn.PIHNN.init`.

        :param method: Either 'he', 'he_scaled', 'exp', see description above.
        :type method: str
        :param beta: Scaling coefficient in the scaled He initialization, :math:`\\beta` coefficient in the Calafà initialization, not used in He initialization.
        :type beta: float
        :param sample: Initial sample :math:`x_0` in the Calafà initialization, not used in the other methods.
        :type sample: :class:`torch.tensor`
        :param gauss: :math:`M_e` coefficient in the Calafà initialization, not used in the other methods.
        :type gauss: int
        """
        if (method=='exp'):
            if(gauss==None):
                gauss = self.n_layers

            x = sample.clone()
            domains = self.dd_partition(x)
            max_size = torch.max(domains.sum(1))
            x = torch.nan*torch.empty(self.n_domains, max_size, dtype=sample.dtype, device=device)
            for d in range(self.n_domains):
                x[d,:domains[d,:].sum()] = sample[domains[d,:]]
                
            for i in range(self.n_layers):
                if (i<gauss):
                    scaling = beta/torch.nanmean(torch.pow(torch.abs(x),2)).detach()
                else:
                    scaling = beta/math.exp(beta)
                self.layers[i].init(scaling)
                y = self.layers[i](x)
                x = self.activation(y)
            return

        if(method=='he'):
            scaling = 1
        elif(method=='he_scaled'):
            scaling = beta
        else:
            raise ValueError("'method' must be either 'exp','he' or 'he_scaled'.")
        for i in range(self.n_layers):
            self.layers[i].init(scaling)



class enriched_PIHNN(DD_PIHNN):
    """
    PIHNN with enrichment for cracks, as introduced in `Calafà et al. [2025] <https://doi.org/10.1016/j.engfracmech.2025.111133>`_.

    It is well-known that traditional solvers and NNs face some difficulties to capture the stress field singularities at the cracks.
    This class hence employs two strategies:

    * Enrichment with **Williams approximation** (Williams, M., “On the stress distribution at the base of a stationary crack”, 1957).
    * Enrichment with **Rice formula** (Rice, J. R., “Mathematical Analysis in the Mechanics of Fracture”, 1968).
    """
    def __init__(self, PDE, units, boundary, material={"lambda": 1, "mu": 1}, activation=torch.exp, has_bias=True):
        """
        :param PDE: Problem to solve, either 'km' or 'km-so'.
        :type PDE: str
        :param units: Number of units at each layer for :math:`NN_0`, e.g., [1,10,10,1].
        :type units: list of int
        :param boundary: Geometry of the domain. Needed for information regarding cracks locations and DD partition.
        :type boundary: :class:`pihnn.geometries.boundary`
        :param material: Properties of the material, dictionary with 'lambda' (first Lamé coefficient), 'mu' (second Lamé coefficient).
        :type material: dict
        :param activation: Activation function, by default the complex exponential.
        :type activation: callable
        :param has_bias: True if the linear layers include bias vectors.
        :type has_bias: bool
        """
        super(enriched_PIHNN, self).__init__(PDE, units, boundary, material, activation, has_bias)
        if PDE not in ['km','km-so']:
            raise ValueError("Enriched PIHNNs can be used only for linear elasticity problems.")
        if boundary.dd_partition is None:
            raise ValueError("Enriched PIHNNs can be used only with DD partitioning.")
        self.enrichment = boundary.enrichment
        self.cracks = boundary.cracks

        sif = []
        for crack in self.cracks:
            for tip in crack.tips:
                sif.append(tip.initial_sif)
        if self.enrichment == "williams":
            self.sif = torch.nn.Parameter(torch.tensor(sif, device=device))

        elif self.enrichment == "rice":
            self.sif = 0*torch.tensor(sif, device=device)
            self.has_crack = torch.zeros([self.n_domains], dtype=torch.bool, device=device)
            self.crack_is_internal = torch.zeros_like(self.has_crack, device=device)
            self.crack_coords = torch.zeros([self.n_domains,1], dtype=torch.complex128, device=device)
            self.crack_angle = torch.zeros([self.n_domains,1], dtype=torch.double, device=device)
            self.crack_a = torch.zeros_like(self.crack_angle, device=device)
            for crack in self.cracks:
                d = crack.rice["domain"]
                self.has_crack[d] = 1
                self.crack_is_internal[d] = crack.rice["is_internal"]
                self.crack_coords[d] = crack.rice["coords"]
                self.crack_angle[d] = crack.rice["angle"]
                self.crack_a[d] = crack.length/2


    def forward(self, z, flat_output=True, real_output=False, force_williams=False):
        """
        Evaluation of the neural network.
        This function calls either :func:`pihnn.nn.enriched_PIHNN.apply_williams` or :func:`pihnn.nn.enriched_PIHNN.apply_rice` based on the setting in :class:`pihnn.geometries.boundary`.
        
        :param z: Input of the network, typically a batch of coordinates from the domain boundary.
        :type z: :class:`torch.tensor`
        :param flat_output: If True, the output of the network is a 1D/flat vector. Otherwise, the output is a 2D tensor where the first dimension is the number of domains and the second dimension
            is the number of points per domain. The second option is necessary for the training of the network while one can simply consider a flat output in other circumstances. 
            Notice that the output is flat only if the input is also flat.
        :type flat_output: bool
        :param real_output: Whether to provide the output in the real-valued representation.
        :type real_output: bool
        :param force_williams: Intended for internal use, it forces the Williams enrichment on the top of the Rice enrichment.
        :type output_mode: bool
        :returns: **phi** (:class:`torch.tensor`) - Output of the network.
        """
        if (len(z.shape)==1):
            domains = self.dd_partition(z)
            z_dd = self.unflatten(z, domains)
        else:
            z_dd = z

        if self.enrichment == "williams":
            phi = super(enriched_PIHNN, self).forward(z_dd)
            phi = phi + self.apply_williams(z_dd)
        elif self.enrichment == "rice":
            phi = self.apply_rice(z_dd)
            if force_williams:
                phi = phi[:2] + self.apply_williams(z_dd)

        if (flat_output and 'domains' in locals()):
            phi = self.flatten(phi, domains)
        
        if real_output:
            return self.apply_real_transformation(z, phi)
        else:
            return phi
        

    def apply_williams(self, z):
        """
        Application of the Williams approximation.

        Namely, the stress field close to a horizontal crack tip at the origin is described by:

        .. math::
            \\begin{cases}
            \\varphi_W(z)= \overline{K}\sqrt{z}, \\\\
            \psi_W(z)= \left(K - \dfrac{\overline{K}}{2}\\right)\sqrt{z},
            \\end{cases}

        where :math:`K=K_I+iK_{II}` is the complex number that combines together the stress intensity factors (SIFs) from mixed mode I-II.
        Here, :math:`K` is a trainable parameter and :math:`\\varphi_W,\\psi_W` are suitably transformed in order to take into account crack tips in any position and direction.
        
        :param z: Input of the network, typically a batch of coordinates from the domain boundary.
        :type z: :class:`torch.tensor`
        :returns: **phi** (:class:`torch.tensor`) - Output of the network.
        """
        phi = torch.zeros((2,)+z.shape,device=device)*1j
        i = 0
        for crack in self.cracks:
            for t in crack.tips:
                d = t.domains
                bcr = t.branch_cut_rotation
                c1 = torch.conj(self.sif[i])
                c2 = (self.sif[i] - torch.conj(self.sif[i])/2)
                rot_sqrt = torch.sqrt(torch.exp(-1j*(bcr+t.angle))*(z[d]-t.coords))
                phi_williams = c1 * torch.exp(1j*(t.angle+bcr/2)) * rot_sqrt
                psi_williams = c2 * torch.exp(1j*(-t.angle+bcr/2)) * rot_sqrt - 0.5 * c1 * torch.conj(t.coords) * torch.exp(-1j*bcr/2) / rot_sqrt
                phi[0,d,:] = phi[0,d,:] + phi_williams/math.sqrt(2*math.pi)
                phi[1,d,:] = phi[1,d,:] + psi_williams/math.sqrt(2*math.pi)
                i+=1
        return phi


    def apply_rice(self, z):
        """
        Application of the Rice formula.

        Namely, a stress-free crack is described by the potentials

        .. math::
            \\begin{cases}
            \\varphi(z)=\sqrt{z-a}\sqrt{z+a}f(z) + g(z), \\\\
            \\omega(z)= \sqrt{z-a}\sqrt{z+a}\hat{f}(z) - \hat{g}(z),
            \\end{cases}
        
        where :math:`\hat{f}(z):=\overline{f(\overline{z})}`, :math:`\omega(z):=z\\varphi'(z) + \psi(z)`, :math:`f,g` are two holomorphic functions and

        .. math::
            \sigma(z) = 
            \\begin{cases}
            \sqrt{z-a}\sqrt{z+a}, & \\text{ for an internal crack at } \{x\in[-a,a], y=0\}, \\\\
            \sqrt{z}, & \\text{ for an open crack at } \{x\in[-\infty,0], y=0\}.
            \\end{cases}

        The method applies the above representation and applies roto-translations.

        :param z: Input of the network, typically a batch of coordinates from the domain boundary.
        :type z: :class:`torch.tensor`
        :returns: **phi** (:class:`torch.tensor`) - Output of the network.
        """
        t = torch.exp(-1j*self.crack_angle)*(z-self.crack_coords)
        phi = super(enriched_PIHNN, self).forward(t) # phi = [f,g]
        phi_h = torch.conj(super(enriched_PIHNN, self).forward(torch.conj(t))) # phi_h = [\check{f},\check{g}]
        sigma = torch.sqrt(t)*(~self.crack_is_internal) + torch.sqrt(t-self.crack_a)*torch.sqrt(t+self.crack_a)*self.crack_is_internal

        if self.PDE == 'km':
            varphi_0 = (phi[0]*sigma+phi[1]) * self.has_crack + phi[0] * (~self.has_crack)
            omega_0 = (phi_h[0]*sigma-phi_h[1]) * self.has_crack + phi[1] * (~self.has_crack)
            varphi_t_0 = derivative(varphi_0, t, holom=True)
            psi_0 = omega_0 - t * varphi_t_0
            # Roto-translations
            varphi = torch.exp(1j*self.crack_angle)*varphi_0
            psi = torch.exp(-1j*self.crack_angle)*psi_0 - torch.conj(self.crack_coords)*varphi_t_0
            varphi_z = varphi_t_0
            return torch.stack([varphi,psi,varphi_z],0) # We keep varphi_z in order to save some computation time

        elif self.PDE == 'km-so':
            varphi_t_0 = (phi[0]/sigma+phi[1]) * self.has_crack + phi[0] * (~self.has_crack)
            omega_t_0 = (phi_h[0]/sigma-phi_h[1]) * self.has_crack + phi[1] * (~self.has_crack)
            varphi_tt_0 = derivative(varphi_t_0, t, holom=True)
            psi_t_0 = omega_t_0 - t * varphi_tt_0 - varphi_t_0
            # Roto-translations
            varphi_z = varphi_t_0
            psi_z = torch.exp(-2j*self.crack_angle)*psi_t_0 - torch.conj(self.crack_coords)*torch.exp(-1j*self.crack_angle)*varphi_tt_0
            varphi_zz = torch.exp(-1j*self.crack_angle)*varphi_tt_0
            return torch.stack([varphi_z,psi_z,varphi_zz],0) # We keep varphi_zz in order to save some computation time