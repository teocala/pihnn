"""Definition of PIHNN networks."""

import torch
import math, warnings
import numpy as np
torch.set_default_dtype(torch.float64) # This also implies default complex128. It applies to all scripts of the library
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Shared with other scripts of the library


class ComplexLinear(torch.nn.Module):
    """
    Base: :class:`torch.nn.Module`

    Extension of :class:`torch.nn.Linear` to complex values. So, a complex linear layer performs the operation
    
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
            self.W = torch.nn.Parameter(torch.empty((stack_features, n_domains, out_features, in_features), dtype=torch.complex128))
        else:
            self.W = torch.nn.Parameter(torch.empty((stack_features, out_features, in_features), dtype=torch.complex128))
        if(self.has_bias):
            if(self.is_dd):
                self.B = torch.nn.Parameter(torch.empty((stack_features, n_domains, out_features, 1), dtype=torch.complex128))
            else:
                self.B = torch.nn.Parameter(torch.empty((stack_features, out_features, 1), dtype=torch.complex128))


    def init(self, scaling):
        """
        Re-initialization of weights and bias.

        Initialization is defined as a scaled complex-valued He initialization (`Trabelsi [2018] <https://arxiv.org/abs/1705.09792>`_):

        .. math::
            \\text{Re}(w),\\text{Im}(w) &\sim \mathcal{N}\left(0,\\frac{\\texttt{scaling}}{2 \\texttt{in_features}}\\right), \\\\
            bias &=0.

        This allows us to easily include the initialization strategy from `Calafà et al. [2024] <https://arxiv.org/abs/2407.01088>`_, Section 3.2.4.
        
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
    Base: :class:`torch.nn.Module`

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
    Base: :class:`torch.nn.Module`

    Main class for the employment of physics-informed holomorphic neural networks (PIHNNs).

    PIHNNs are able to solve 4 types of problems:

    * Laplace problem ('laplace'): 
        :math:`\\nabla^2u=0 \\Leftrightarrow u=\\text{Re}(\\varphi)` where :math:`\\varphi` is a holomorphic function.
    * Biharmonic problem with Goursat representation ('biharmonic'): 
        :math:`\\nabla^4u=0 \\Leftrightarrow u=\\text{Re}((x-iy)\\varphi + \psi)` where :math:`\\varphi,\psi` are holomorphic functions.
    * Linear elasticity with Kolosov-Muskhelishvili representation, standard ('km'): 
        Stresses and displacements can be obtained from :math:`\\varphi,\psi`, see `Calafà et al. [2024] <https://arxiv.org/abs/2407.01088>`_, Section 3.2.2.
    * Linear elasticity with Kolosov-Muskhelishvili representation, stress-only ('km-so'): 
        Stresses can be obtained from :math:`\\varphi,\psi`, see `Calafà et al. [2024] <https://arxiv.org/abs/2407.01088>`_, Section 3.2.2.
    
    The output of the network is therefore the scalar function :math:`\\varphi_{NN}\\approx\\varphi` in the Laplace problem. 
    Instead, for the other problems the PIHNN is composed by 2 stacked networks :math:`\\varphi_{NN},\psi_{NN}\\approx\\varphi,\psi`.
    """
    def __init__(self, PDE, units, material={"lambda": 1, "mu": 1}, activation=torch.exp, has_bias=True):
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
        """
        super(PIHNN, self).__init__()
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
        self.layers = torch.nn.ModuleList()
        if (PDE == 'km-so'):
            warnings.warn("You are using the 'stress-only' configuration. Please ensure that boundary conditions are only of stress type.")

        for i in range(self.n_layers):
            self.layers.append(ComplexLinear(self.n_outputs, units[i], units[i+1], has_bias=has_bias))


    def forward(self, z):
        """
        Forward step, i.e., compute for :math:`j=1,2`:

        .. math::
            \mathcal{L}_{N,j} \circ \phi \circ \mathcal{L}_{N-1,j} \circ \phi \dots \circ \mathcal{L}_{1,j} (z)

        where :math:`z` is the input, :math:`\phi` the activation function and :math:`\{\mathcal{L}_{i,j}\}` the complex linear layers (:class:`pihnn.nn.ComplexLinear`) for each layer :math:`i` and stacked network :math:`j`.

        :param z: Input of the network, typically a batch of coordinates from the domain boundary.
        :type z: :class:`torch.tensor`
        :returns: **phi** (:class:`torch.tensor`) - Output of the network. As mentioned above, it has the same shape of the input for the Laplace problem but double size for the other problems.
        """
        phi = self.layers[0](z)
        for i in range(1, self.n_layers):
            phi = self.layers[i](self.activation(phi))
        return phi.squeeze(1)


    def initialize_weights(self, method, beta=0.5, sample=None, gauss=None):
        """
        Initialization of PIHNNs. Implemented methods:

        * Complex-valued He initialization (`Trabelsi [2018] <https://arxiv.org/abs/1705.09792>`_):
            :math:`\\text{Re}(w),\\text{Im}(w)\sim \mathcal{N}\left(0,\\frac{1}{2 \\texttt{in_features}}\\right), \hspace{3mm} bias=0`.
        * Scaled complex-valued He initialization:
            :math:`\\text{Re}(w),\\text{Im}(w)\sim \mathcal{N}\left(0,\\frac{\\texttt{scaling}}{2 \\texttt{in_features}}\\right), \hspace{3mm} bias=0`.
        * PIHNNs ad-hoc initialization with exponential activations:
            See `Calafà et al. [2024] <https://arxiv.org/abs/2407.01088>`_, Section 3.2.4.

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



class DD_PIHNN(PIHNN):
    """
    Base: :class:`torch.nn.Module`

    Domain-decomposition physics-informed holomorphic neural networks (DD-PIHNNs).

    DD-PIHNNs have been introduced in `Calafà et al. [2024] <https://arxiv.org/abs/2407.01088>`_, Section 4.3, to solve problems on multiply-connected domains.
    The structure is similar to :class:`pihnn.nn.PIHNN` but includes multiple stacked networks, each one corresponding to each function :math:`\\varphi,\psi` and each domain.
    """
    def __init__(self, PDE, units, boundary, material={"lambda": 1, "mu": 1}, activation=torch.exp, has_bias=True):
        """
        :param PDE: Problem to solve, either 'laplace', 'biharmonic', 'km' or 'km-so'.
        :type PDE: str
        :param units: List containing number of units at each layer, e.g., [1,10,10,1].
        :type units: list of int
        :param boundary: Geometry of the domain, necessary for information regarding domain splitting.
        :type boundary: :class:`pihnn.geometries.boundary`
        :param material: Properties of the material, dictionary with 'lambda' (first Lamé coefficient), 'mu' (second Lamé coefficient).
        :type material: dict
        :param activation: Activation function, by default the complex exponential.
        :type activation: callable
        :param has_bias: True if the linear layers include bias vectors.
        :type has_bias: bool
        """
        super(DD_PIHNN, self).__init__(PDE, units, material, activation, has_bias)
        if boundary.dd_partition is None:
            raise ValueError("Boundary must be initialized with well-defined 'dd_partition' in order to create DD-PIHNNs.")
        self.dd_partition = boundary.dd_partition
        self.n_domains = boundary.n_domains
        self.layers = torch.nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(ComplexLinear(self.n_outputs, units[i], units[i+1], self.n_domains, has_bias=has_bias))
        if (PDE == 'km-so'):
            raise ValueError("'stress-only' configuration cannot be applied to DD-PIHNNs.")


    def forward(self, z, flat_output=True):
        """
        Forward step, i.e., compute for :math:`j=1,2`:

        .. math::
            \mathcal{L}_{N,j,d} \circ \phi \circ \mathcal{L}_{N-1,j,d} \circ \phi \dots \circ \mathcal{L}_{1,j,d} (z)

        where :math:`z` is the input, :math:`\phi` the activation function, :math:`d\in \mathbb{N}` the domain to which :math:`z` belongs and :math:`\{\mathcal{L}_{i,j,d}\}` the complex linear layers (:class:`pihnn.nn.ComplexLinear`) for each layer :math:`i` and stacked network :math:`(j,d)`.

        :param z: Input of the network, typically a batch of coordinates from the domain boundary.
        :type z: :class:`torch.tensor`
        :param flat_output: If True, the output of the network is a 1D/flat vector. Otherwise, the output is a 2D tensor where the first dimension is the number of domains and the second dimension
            is the number of points per domain. The second option is necessary for the training of the network while one can simply consider a flat output in other circumstances. 
            Notice that the output is flat only if the input is also flat.
        :type flat_output: bool
        :returns: **phi** (:class:`torch.tensor`) - Output of the network. It has the same shape of the input for the Laplace problem but double size for the other problems.
        """

        if (len(z.shape)==1): # Flat input
            z_old = z.clone()
            domains = self.dd_partition(z)
            max_size = torch.max(domains.sum(1))
            z = torch.empty(self.n_domains, max_size, dtype=z_old.dtype)
            for d in range(self.n_domains): # We do similarly to boundary.extract_points_dd
                z[d,:domains[d,:].sum()] = z_old[domains[d,:]]

        phi = self.layers[0](z)
        for i in range(1, self.n_layers):
            phi = self.layers[i](self.activation(phi))

        if (flat_output and 'z_old' in locals()):
            flat_phi = torch.empty((self.n_outputs,)+z_old.shape, dtype=z_old.dtype)
            for d in range(self.n_domains): # We do the inverse of the previous operation
                flat_phi[:,domains[d,:]] = phi[:,d,0,:domains[d,:].sum()]
            return flat_phi
        
        return phi.squeeze(2)


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
            x = torch.nan*torch.empty(self.n_domains, max_size, dtype=sample.dtype)
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