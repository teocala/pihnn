"""
Geometry entities for the definition of boundary-value problems.

The class :class:`pihnn.geometries.boundary` contains all the information regarding the geometry and the boundary conditions applied on it. 
It is composed by a sequence of edges/curves, each one corresponding to one type of BC.
"""
import torch
import inspect, warnings
import pihnn.nn as nn
import pihnn.bc as bc
import pihnn.utils as utils


class boundary():
    """Main class for the definition of the domain boundary and boundary conditions (BCs)."""
    def __init__(self, curves, np_train, np_test, dd_partition=None, enrichment=None):
        """
        :param curves: List of :class:`pihnn.geometry.curve` which defines the domain boundary.
        :type curves: list
        :param np_train: Number of training points.
        :type np_train: int
        :param np_test: Number of test points.
        :type np_test: int
        :param dd_partition: Domain decomposition rule, only for DD-PIHNNs.
        :type dd_partition: callable
        :param enrichment: Crack enrichment method, either "williams" or "rice". Leave it to None for no enrichment.
        :type enrichment: string

        An example of domain splitting for partitioning into the 4 quadrants is as follows

        .. code-block:: python

            def dd_partition (x,y): # Domain decomposition partition
                return torch.stack([
                (x>=-1e-10) & (y>=-1e-10),
                (x<=1e-10) & (y>=-1e-10),
                (x<=1e-10) & (y<=1e-10),
                (x>=-1e-10) & (y<=1e-10),
                ])

        .. note:: dd_partition must be in exactly the same format as the example: it must return a tensor with an additional dimension at position 0 and each domain must contain also shared interfaces (this is the reason for 1e-10).
        """
        self.curves = curves
        self.np_train = np_train
        self.np_test = np_test
        self.enrichment = enrichment
        self.stress_only = True
        self.multi_connected = False
        self.dd_partition = None
        self.square = curves[0].square
        self.length = 0.
        self.bc_types = []
        self.bc_names = []
        self.cracks = []

        length_fixed_sampling = 0.
        cumulative_sampling_ratio = 0.

        for curve in self.curves:
            if not isinstance(curve.bc_type, bc.stress_bc): # I.e., we also need to compute displacements
                self.stress_only = False
            if isinstance(curve.bc_type, bc.interface_bc): # I.e., domain decomposition
                self.multi_connected = True  
            self.length += curve.length

            self.square[0] = torch.minimum(self.square[0], curve.square[0])
            self.square[1] = torch.maximum(self.square[1], curve.square[1])
            self.square[2] = torch.minimum(self.square[2], curve.square[2])
            self.square[3] = torch.maximum(self.square[3], curve.square[3])

            if curve.bc_name not in self.bc_names:
                self.bc_names.append(curve.bc_name)
                self.bc_types.append(curve.bc_type)

            if len(curve.tips) != 0:
                curve.on_boundary = False
                if enrichment == "rice":
                    curve.sampling_ratio = 0.
                    self.curves.remove(curve)
                    if not isinstance(curve, line):
                        raise ValueError("Rice method only works for straight line cracks")
                self.cracks.append(curve)

            if curve.sampling_ratio is not None:
                length_fixed_sampling += curve.length
                cumulative_sampling_ratio += curve.sampling_ratio

        if cumulative_sampling_ratio >= 1 + 1e-10:
            raise ValueError("Sum of 'sampling_ratio' in curves must be between 0 and 1.")
        for curve in self.curves:
            if curve.sampling_ratio is None:
                curve.sampling_ratio = curve.length*(1-cumulative_sampling_ratio)/(self.length-length_fixed_sampling)

        self.adjust_orientation()

        if self.multi_connected and dd_partition is None:
            raise ValueError("The domain is multiply-connected and is missing a domain partitioning.")
        if not self.multi_connected and dd_partition is not None:
            raise ValueError("Domain partitioning is not accepted for simply-connected domains.") 
        
        if self.cracks != [] and dd_partition is None:
                dd_partition = lambda z: torch.ones([1,z.shape[0]], dtype=torch.bool) # Trivial dd_partition
        
        self.n_domains = 1
        if dd_partition is not None:
            if len(inspect.getfullargspec(dd_partition)[0]) == 1:
                self.dd_partition = dd_partition
            elif len(inspect.getfullargspec(dd_partition)[0]) == 2:
                self.dd_partition = lambda z: dd_partition(z.real, z.imag)
            else:
                raise ValueError("'dd_partition' must be a function of two real arguments or a single complex argument.")
            output = self.dd_partition(torch.tensor([0.j]))
            if output.shape[1] != 1:
                raise ValueError("'dd_partition' should return a tensor of shape [D,N], where D is the number of domains and N the number of points.")  
            self.n_domains = output.shape[0]            

        self.points_train, self.normals_train, self.bc_idxs_train, self.bc_values_train = self.extract_points(self.np_train)
        self.points_test, self.normals_test, self.bc_idxs_test, self.bc_values_test = self.extract_points(self.np_test)  

        if self.multi_connected or dd_partition is not None:
            self.extract_points_dd()

        if self.cracks == [] and enrichment in ["williams","rice"]:
            raise ValueError(f"Enrichment is {enrichment} but no tips have been added.")
        if self.cracks != []:
            if enrichment not in ["williams","rice"]:
                raise ValueError("Enrichment method must be either 'williams' or 'rice'.")
            else:
                self.initialize_cracks()

        self.check_consistency()


    def extract_points(self, N):
        """
        Generation of a new batch of points from the boundary.

        :param N: Number of points to generate
        :type N: int
        :returns: 
            - **points** (:class:`torch.tensor`) - Coordinates of the extracted points.
            - **normals** (:class:`torch.tensor`) - Coordinates of the boundary outward vectors at the points locations.
            - **bc_idxs** (:class:`torch.tensor`) - Type of BC at the corresponding points.
            - **bc_values** (:class:`torch.tensor`) - Assigned BC value at the corresponding points.
        """
        n0 = 0
        points = 1j*torch.empty(N).to(nn.device)
        normals = 1j*torch.empty(N).to(nn.device)
        bc_idxs = torch.empty(N, dtype=torch.int8).to(nn.device)
        bc_values = 1j*torch.empty(N).to(nn.device)

        for curve in self.curves:
            if (curve == self.curves[-1]):
                n = N - n0 # Divisions could lead to leftover points, so in this way we are sure that all the points are considered.
            else:
                n = int(N * curve.sampling_ratio)
            if (curve.ref_loc=="center"):
                seed = 0.5 + (0.5-torch.bernoulli(0.5*torch.ones(n)))*torch.pow(torch.rand(n), 1./curve.order)
            elif (curve.ref_loc=="start"):
                seed = torch.pow(torch.rand(n), curve.order)
            elif (curve.ref_loc=="end"):
                seed = torch.pow(torch.rand(n), 1./curve.order)
            points[n0:n0+n], normals[n0:n0+n] = curve.map_point(seed.to(nn.device))
            bc_idxs[n0:n0+n] =  self.bc_names.index(curve.bc_name)
            bc_values[n0:n0+n] = curve.bc_value(points[n0:n0+n])
            n0 += n
        return points, normals, bc_idxs, bc_values


    def extract_points_dd(self):
        """
        Training and test points generated from :func:`extract_points()` are copied internally in an additional format that is necessary for the training of DD-PIHNNs.

        In particular, if :math:`N` is the number of training/test points and :math:`D` the number of domains, we have the transformation

        .. math::
            P_f \in \mathbb{C}^{N} \\rightarrow P_d \in \mathbb{C}^{D,N_D}

        where :math:`N_D` is the maximum number of points in a domain (approximately :math:`N/D` if the domains are all similar). 
        In particular, :math:`p\in P_f` is copied into :math:`P_d` at the :math:`i`-th row if :math:`p` belongs to the :math:`i`-th domain. 

        Notice that the division/splitting is almost never exact, therefore there are some elements in :math:`P_d` that are empty and therefore ignored.

        Together with :math:`P_d` for both training and test points, the method provides the mask used to ignore leftover points and another tensor used to identify twin points on interfaces.
        """
        if not hasattr(self, 'points_train'):
            raise ValueError("The method can be used only after training and test points are generated through extract_points()")
        if self.dd_partition == None:
            raise ValueError("'dd_partition' must be initialized before extracting the points in DD.")
        
        iibc = self.bc_names.index("interface_bc") if "interface_bc" in self.bc_names else -1
        domains = self.dd_partition(self.points_train)
        n_domains = domains.shape[0]
        max_size = torch.max(domains.sum(1)) # Maximum number of points in a domain

        self.points_train_dd = torch.nan*torch.empty(n_domains, max_size, dtype=self.points_train.dtype, device=nn.device)
        self.normals_train_dd = torch.empty(n_domains, max_size, dtype=self.normals_train.dtype, device=nn.device)
        self.bc_idxs_train_dd = torch.empty(n_domains, max_size, dtype=self.bc_idxs_train.dtype, device=nn.device)
        self.bc_values_train_dd = torch.empty(n_domains, max_size, dtype=self.bc_values_train.dtype, device=nn.device)

        for d in range(n_domains):
            self.points_train_dd[d,:domains[d,:].sum()] = self.points_train[domains[d,:]]
            self.normals_train_dd[d,:domains[d,:].sum()] = self.normals_train[domains[d,:]]
            self.bc_idxs_train_dd[d,:domains[d,:].sum()] = self.bc_idxs_train[domains[d,:]]
            self.bc_values_train_dd[d,:domains[d,:].sum()] = self.bc_values_train[domains[d,:]]

        self.mask_train_dd = torch.isnan(self.points_train_dd)

        self.twins_train_dd = torch.empty((4,0), dtype=torch.int64) # Four rows: domain of left point, domain of right point, index of left point, index of right point.
        for i in range(n_domains):
            for j in range(i+1, n_domains):
                ij_size = torch.logical_and(self.dd_partition(self.points_train_dd[i,:])[j,:], self.bc_idxs_train_dd[i,:]==iibc).sum()
                n0 = self.twins_train_dd.shape[1]
                self.twins_train_dd = torch.cat((self.twins_train_dd, torch.empty((4, ij_size), dtype=torch.int64)), dim=1)
                self.twins_train_dd[0,n0:] = i
                self.twins_train_dd[1,n0:] = j
                self.twins_train_dd[2,n0:] = torch.where(self.dd_partition(self.points_train_dd[i,:])[j,:] & (self.bc_idxs_train_dd[i,:]==iibc))[0]
                self.twins_train_dd[3,n0:] = torch.where(self.dd_partition(self.points_train_dd[j,:])[i,:] & (self.bc_idxs_train_dd[j,:]==iibc))[0]


        # Repeat the same for test points
        domains = self.dd_partition(self.points_test)
        n_domains = domains.shape[0]
        max_size = torch.max(domains.sum(1))

        self.points_test_dd = torch.nan*torch.empty(n_domains, max_size, dtype=self.points_test.dtype, device=nn.device)
        self.normals_test_dd = torch.empty(n_domains, max_size, dtype=self.normals_test.dtype, device=nn.device)
        self.bc_idxs_test_dd = torch.empty(n_domains, max_size, dtype=self.bc_idxs_test.dtype, device=nn.device)
        self.bc_values_test_dd = torch.empty(n_domains, max_size, dtype=self.bc_values_test.dtype, device=nn.device)

        for d in range(n_domains):
            self.points_test_dd[d,:domains[d,:].sum()] = self.points_test[domains[d,:]]
            self.normals_test_dd[d,:domains[d,:].sum()] = self.normals_test[domains[d,:]]
            self.bc_idxs_test_dd[d,:domains[d,:].sum()] = self.bc_idxs_test[domains[d,:]]
            self.bc_values_test_dd[d,:domains[d,:].sum()] = self.bc_values_test[domains[d,:]]

        self.mask_test_dd = torch.isnan(self.points_test_dd)       

        self.twins_test_dd = torch.empty((4,0), dtype=torch.int64)
        for i in range(n_domains):
            for j in range(i+1, n_domains):
                ij_size = torch.logical_and(self.dd_partition(self.points_test_dd[i,:])[j,:], self.bc_idxs_test_dd[i,:]==iibc).sum()
                n0 = self.twins_test_dd.shape[1]
                self.twins_test_dd = torch.cat((self.twins_test_dd, torch.empty((4, ij_size), dtype=torch.int64)), dim=1)
                self.twins_test_dd[0,n0:] = i
                self.twins_test_dd[1,n0:] = j
                self.twins_test_dd[2,n0:] = torch.where(self.dd_partition(self.points_test_dd[i,:])[j,:] & (self.bc_idxs_test_dd[i,:]==iibc))[0]
                self.twins_test_dd[3,n0:] = torch.where(self.dd_partition(self.points_test_dd[j,:])[i,:] & (self.bc_idxs_test_dd[j,:]==iibc))[0]

        # Finally, we replace the "NaN" for ignored points
        # Precise values are meaningless since they are not really used and the only reason is to avoid NaN or Inf
        self.points_train_dd[self.mask_train_dd] = 0.123456789j # Any fictitious value outside the "true" boundary 
        self.normals_train_dd[self.mask_train_dd] = 1.j
        self.bc_idxs_train_dd[self.mask_train_dd] = 0
        self.bc_values_train_dd[self.mask_train_dd] = 0.
        self.points_test_dd[self.mask_test_dd] = 0.123456789j
        self.normals_test_dd[self.mask_test_dd] = 1.j
        self.bc_idxs_test_dd[self.mask_test_dd] = 0
        self.bc_values_test_dd[self.mask_test_dd] = 0.


    def __call__(self, dataset, dd=False):
        """
        It returns the currently saved boundary points (generated through :func:`extract_points`).

        :param dataset: 'training' or 'test' option.
        :type dataset: str
        :param dd: If True, it returns the boundary points in the domain decomposition format (see :func:`extract_points_dd`)
        :type dd: bool
        :returns: 
            - **points** (:class:`torch.tensor`) - Coordinates of the extracted points.
            - **normals** (:class:`torch.tensor`) - Coordinates of the boundary outward vectors at the points locations.
            - **bc_idxs** (:class:`torch.tensor`) - Type of BC at the corresponding points.
            - **bc_values** (:class:`torch.tensor`) - Assigned BC value at the corresponding points.
        """
        if not hasattr(self, 'points_train'):
            raise ValueError("You have to generate training and test points through extract_points() before calling the method.")
        if(not hasattr(self, 'points_train_dd') and dd):
            raise ValueError("You have to generate training and test points through extract_points_dd() before calling the method with the dd specifier.")

        if (dataset=="training" and not dd):
            return self.points_train, self.normals_train, self.bc_idxs_train, self.bc_values_train
        elif (dataset=="test" and not dd):
            return self.points_test, self.normals_test, self.bc_idxs_test, self.bc_values_test
        elif (dataset=="training" and dd):
            return self.points_train_dd, self.normals_train_dd, self.bc_idxs_train_dd, self.bc_values_train_dd, self.mask_train_dd, self.twins_train_dd
        elif (dataset=="test" and dd):
            return self.points_test_dd, self.normals_test_dd, self.bc_idxs_test_dd, self.bc_values_test_dd, self.mask_test_dd, self.twins_test_dd
        else:
            raise ValueError("'dataset' must be either 'training' or 'test'.")

    
    def is_inside(self, points):
        """
        Verifies whether some coordinates are inside or outside the boundary. It employs the 'ray casting algorithm'.

        :param points: Batch of points to inspect.
        :type points: :class:`torch.tensor`
        :returns: **inside** (:class:`torch.tensor`) - For each input point, True if it is inside and False if it outside.
        """
        inside = torch.zeros(points.shape, dtype=torch.int8)
        for curve in self.curves:
            if curve.on_boundary:
                inside += curve.intersect_ray(points)
        return torch.remainder(inside, 2)


    def adjust_orientation(self):
        """
        This function is called internally to adjust the sign of the normals to the boundary. It auto-detects if the normals are inward and make them outward.
        """
        for curve in self.curves:
            p, normal = curve.map_point(0.5) # Point in the middle of the curve.
            p = p + 1e-5*normal # This can be either slightly inside or slightly outside the domain.
            if(self.is_inside(p)): # I.e., the normal vector is inward instead of outward.
                curve.orientation = -1


    def check_consistency(self):
        """
        Checks if boundary is formed by a well-defined closed loop, throws a warning otherwise.
        """
        for j, curve1 in enumerate(self.curves):
            if not curve1.check_consistency:
                continue
            if torch.abs(curve1.P1-curve1.P2) < 1e-6: # The curve is already closed
                continue
            cnt1 = 0
            cnt2 = 0
            for curve2 in self.curves:
                cnt1 += curve2.is_inside(curve1.P1)
                cnt2 += curve2.is_inside(curve1.P2)
            if(cnt1==1):
                warnings.warn(f"Boundary is not closed, 1st edge of {utils.ordinal_number(j+1)} curve is not connected to any other edge.", UserWarning) 
            if(cnt2==1):
                warnings.warn(f"Boundary is not closed, 2nd edge of {utils.ordinal_number(j+1)} curve is not connected to any other edge.", UserWarning) 


    def initialize_cracks(self):
        """
        Initialization of pihnn.geometries.crack_line within a boundary geometry.

        Specifically, the method detects whether the crack is interior or open and which domain it belongs to.
        """
        if self.dd_partition == None:
            raise ValueError("Cracks can be handled only with DD-PIHNNs.")

        if len(self.cracks) > self.n_domains:
            raise ValueError("Enrichment works with maximum 1 crack per subdomain.")        

        if self.enrichment == "williams":
            for crack in self.cracks:
                for tip in crack.tips:
                    left_point = tip.coords + 1*(1j*tip.norm)
                    right_point = tip.coords + 1*(-1j*tip.norm)
                    domain_left = self.dd_partition(left_point).squeeze(-1).nonzero().item()
                    domain_right = self.dd_partition(right_point).squeeze(-1).nonzero().item()
                    tip.domains = [domain_left, domain_right]
                    if domain_left == domain_right:
                        raise ValueError(f"Crack tip at coordinates {tip.coords} is entirely in domain {domain_left} but must be shared between 2 subdomains.")

        elif self.enrichment == "rice":
            for crack in self.cracks:
                for tip in crack.tips:
                    tip.domains = self.dd_partition(tip.coords).squeeze(-1).nonzero().item()
                    tip.branch_cut_rotation = 0*tip.branch_cut_rotation[0]
                if len(crack.tips)==1:
                    crack.rice = {
                        "is_internal": False,
                        "coords": crack.tips[0].coords,
                        "angle": crack.tips[0].angle,
                        "domain": crack.tips[0].domains
                        }
                elif len(crack.tips)==2:
                    crack.rice = {
                        "is_internal": True,
                        "coords": (crack.tips[0].coords + crack.tips[1].coords)/2,
                        "angle": crack.tips[0].angle,
                        "domain": crack.tips[0].domains
                        }
                else:
                    raise ValueError("Each can crack can have at most 2 crack tips.")

        else:
            raise ValueError("'enrichment' must be either 'williams' or 'rice'.")



class curve():
    """
    Portion of boundary, identified with a shape and a type of boundary condition (BC). 
    """
    def __init__(self, bc_type, bc_value=0, order=1, ref_loc="center", check_consistency=True, on_boundary=True, sampling_ratio=None):
        """
        :param bc_type: Type of boundary condition to assign to the curve (see above).
        :type bc_type: int
        :param bc_value: Value assigned at the boundary condition. It can be either a constant or a variable value defined through a function.
        :type bc_value: callable / complex
        :param order: Order of sampling refinement at the edges, 1 for no refinement (uniform distribution). For example, if the curve is :math:`[0,1]` and we want to refine on the right side, then the sampled points are distributed as :math:`u^{1/order}`, where :math:`u \sim \mathcal{U}(0,1)` and :math:`order\ge1`.
        :type order: float
        :param ref_loc: Where to apply refinement. 'center' for symmetric refinement (i.e., both edges), 'start' only at the first edge and 'end' only at the second edge.
        :type ref_loc: str
        :param check_consistency: If False, the current curve is excluded from the :func:`pihnn.geometries.boundary.check_consistency` check.
        :type check_consistency: bool
        :param on_boundary: Set to False if the current curve is not really part of the domain boundary.
        :type on_boundary: bool
        :param sampling_ratio: Number of sampled points on the curve with respect to the total sampled points on the boundary. By default, it's the ratio between the length curve and the boundary perimeter.
        :type np_train: float
        """
        self.length = None
        self.square = None
        self.orientation = 1
        self.P1 = None # First edge.
        self.P2 = None # Second edge.
        self.bc_type = bc_type
        self.bc_name = bc_type.__class__.__name__
        self.bc_value = utils.get_complex_function(bc_value)
        self.check_consistency = check_consistency
        self.on_boundary = on_boundary
        self.sampling_ratio = sampling_ratio
        self.tips = []

        if isinstance(bc_type, bc.interface_bc):
            self.on_boundary = False

        if isinstance(order, (int,float)):
            if order > 1e-10:
                self.order = order
            else:
                raise ValueError("'order' must be a positive number.")
        else:
            raise ValueError("'order' must be a real number.")

        if (ref_loc in ['center','start','end']):
            self.ref_loc = ref_loc # "center" if refinement is symmetrical, "start" if it is applied to beginning of curve, "end" if it is applied towards the end of curve.
        else:
            raise ValueError("'ref_loc' must be either 'center', 'start' or 'end'.")
        

    def map_point(self, s):
        """
        Parametrization of the curve, i.e., the bijective function :math:`[0,1] \\rightarrow \gamma \subset \mathbb{C}`.

        :param s: Value(s) in [0,1], typically obtained from a pseudo-random generator.
        :type s: float/:class:`torch.tensor`
        :returns: 
            - **point** (:class:`torch.tensor`) - Coordinates of the mapped point(s).
            - **normal** (:class:`torch.tensor`) - Coordinates of the boundary outward vectors at the mapped point(s).
        """
        raise NotImplementedError


    def is_inside(self, points, tol=1e-6):
        """
        Verifies whether a point lies on the curve, considering a tolerance.

        :param points: Batch of points to inspect.
        :type points: :class:`torch.tensor`
        :param tol: Tolerance for the measure of distance.
        :type tol: float
        :returns: **inside** (:class:`torch.tensor`) - For each input point, True if it is inside and False if it outside.
        """
        raise NotImplementedError


    def intersect_ray(self, points):
        """ 
        Verifies whether a point is on the left of the curve, i.e., if the horizontal leftward straight line starting from the point intersects the curve.
        This function is used for the 'ray casting algorithm' in :func:`pihnn.geometries.boundary.is_inside`.

        :param points: Batch of points to inspect.
        :type points: :class:`torch.tensor`
        :returns: **inside** (:class:`torch.tensor`) - For each input point, True if it is inside and False if it outside.
        """
        raise NotImplementedError


    def add_crack_tip(self, tip_side=0, branch_cut_rotation=[0.1,-0.1], initial_sif=0.):
        """
        Add crack tip for enriched PIHNNs.
        :param tip_side: Which side to consider for the crack tip, 0 is the starting point of the curve, 1 the end.
        :type tip_side: bool
        :param branch_cut_rotation: Rotation of the square root branch cuts with respect to the domain on the left and on the right of the crack, respectively. In radiants and anti-clockwise. Only used for Williams enrichment.
        :type branch_cut_rotation: list of floats
        :param initial_sif: Initial values for the trainable stress intensity factors :math:`K_I` and :math:`K_{II}`. Only used for Williams enrichment.
        :type initial_sif: int/float/complex/list/tuple/tensor
        """
        if not tip_side in [0,1]:
            raise ValueError("tip_side must be a boolean value.")

        close_tip_side = 1e-8
        if tip_side:
            close_tip_side = 1. - close_tip_side

        z, n = self.map_point(torch.tensor([float(tip_side)], device=nn.device))
        zc, _ = self.map_point(torch.tensor([float(close_tip_side)], device=nn.device))
        
        n *= 1.j # Rotation of 90 degrees
        sign = (z-zc).real*n.real + (z-zc).imag*n.imag # Scalar product
        if sign < 0:
            n *= -1
  
        self.tips.append(crack_tip(z, n, branch_cut_rotation, initial_sif))


class line(curve): # Straight line, anti-clockwise with respect to boundary
    """
    Straight line between :math:`P_1,P_2 \in \mathbb{C}`, defined as:

    .. math::
        \gamma:=\{z \in \mathbb{C}: z=tP_1 + (1-t)P_2 \\text{ for some } t\in[0,1]\}.

    """
    def __init__(self, P1, P2, bc_type, bc_value=0, order=1, ref_loc="center", check_consistency=True, on_boundary=True, sampling_ratio=None):
        """
        :param P1: First edge of the line.
        :type P1: int/float/complex/list/tuple/:class:`torch.tensor`
        :param P2: Second edge of the line.
        :type P2: int/float/complex/list/tuple/:class:`torch.tensor`
        :param bc_type: Type of boundary condition to assign to the curve (see above).
        :type bc_type: str
        :param bc_value: Values assigned at the boundary condition. It can be either a constant or a variable value defined through a function.
        :type bc_value: callable / complex
        :param order: Order of sampling refinement at the edges, 1 for no refinement (uniform distribution). For example, if the curve is :math:`[0,1]` and we want to refine on the right side, then the sampled points are distributed as :math:`u^{1/order}`, where :math:`u \sim \mathcal{U}(0,1)` and :math:`order\ge1`.
        :type order: float
        :param ref_loc: Where to apply refinement. 'center' for symmetric refinement (i.e., both edges), 'start' only at the first edge and 'end' only at the second edge.
        :type ref_loc: str
        :param check_consistency: If False, the current curve is excluded from the :func:`pihnn.geometries.boundary.check_consistency` check.
        :type check_consistency: bool
        :param on_boundary: Set to False if the current curve is not really part of the domain boundary.
        :type on_boundary: bool
        :param sampling_ratio: Number of sampled points on the curve with respect to the total sampled points on the boundary. By default, it's the ratio between the curve length and the boundary perimeter.
        :type sampling_ratio: float
        """
        super(line, self).__init__(bc_type, bc_value, order, ref_loc, check_consistency, on_boundary, sampling_ratio)
        self.P1 = utils.get_complex_input(P1)
        self.P2 = utils.get_complex_input(P2)
        self.length = torch.abs(self.P2-self.P1)
        self.square = [torch.minimum(self.P1.real, self.P2.real), torch.maximum(self.P1.real, self.P2.real), torch.minimum(self.P1.imag, self.P2.imag), torch.maximum(self.P1.imag, self.P2.imag)]


    def map_point(self, s): # s in [0,1]
        point = self.P1 + s * (self.P2 - self.P1)
        normal = self.orientation * 1.j * (self.P2-self.P1) / self.length + 0*point # Clockwise or anti-clockwise rotation of 90 degrees
        return point, normal
    

    def is_inside(self, points, tol=1e-6):
        dir1 = points - self.P1
        dir2 = self.P2 - self.P1
        condition_1 = torch.abs(dir1) <= torch.abs(dir2) + tol # The point is between P1, P2
        condition_2 = torch.abs(torch.imag(dir1/dir2)) < tol # dir1 and dir2 are parallel, i.e. the point is along the segment
        return condition_1 & condition_2


    def intersect_ray(self, points):
        output = torch.ones(points.shape, dtype=torch.bool)
        output[points.imag>torch.max(self.P1.imag,self.P2.imag)+1e-10] = False
        output[points.imag<torch.min(self.P1.imag,self.P2.imag)-1e-10] = False
        dir1 = 1.j * (self.P2 - self.P1) # Orthogonal vector to the line.
        if (dir1.real < 0):
            dir1 *= -1
        dir2 = points - self.P1
        output[dir1.real*dir2.real+dir1.imag*dir2.imag>0] = False
        return output



class arc(curve): # Arc of circumference, anti-clockwise with respect to boundary
    """
    Arc of circle defined as:

    .. math::
        \gamma:=\{z \in \mathbb{C}: z= c + re^{i\\theta}, \min(\\theta_1,\\theta_2) \le \\theta \le \max(\\theta_1,\\theta_2) \},

    where :math:`c\in\mathbb{C}` is the center of the circle, :math:`r\in\mathbb{R}` the radius and :math:`\\theta_1,\\theta_2\in\mathbb{R}` the angles corresponding to the two ends.

    """
    def __init__(self, center, radius, theta1, theta2, bc_type, bc_value=0, order=1, ref_loc="center", check_consistency=True, on_boundary=True, sampling_ratio=None):
        """
        :param center: Center of the circle.
        :type center: int/float/complex/list/tuple/:class:`torch.tensor`
        :param radius: Radius of the circle.
        :type radius: float
        :param theta1: Angle corresponding to the first end (in radians, moving anti-clockwise along the domain boundary).
        :type theta1: float
        :param theta2: Angle corresponding to the second end (in radians, moving anti-clockwise along the domain boundary).
        :type theta2: float
        :param bc_type: Type of boundary condition to assign to the curve (see above).
        :type bc_type: str
        :param bc_value: Values assigned at the boundary condition. It can be either a constant or a variable value defined through a function.
        :type bc_value: callable / complex
        :param order: Order of sampling refinement at the edges, 1 for no refinement (uniform distribution). For example, if the curve is :math:`[0,1]` and we want to refine on the right side, then the sampled points are distributed as :math:`u^{1/order}`, where :math:`u \sim \mathcal{U}(0,1)` and :math:`order\ge1`.
        :type order: float
        :param ref_loc: Where to apply refinement. 'center' for symmetric refinement (i.e., both edges), 'start' only at the first edge and 'end' only at the second edge.
        :type ref_loc: str
        :param check_consistency: If False, the current curve is excluded from the :func:`pihnn.geometries.boundary.check_consistency` check.
        :type check_consistency: bool
        :param on_boundary: Set to False if the current curve is not really part of the domain boundary.
        :type on_boundary: bool
        :param sampling_ratio: Number of sampled points on the curve with respect to the total sampled points on the boundary. By default, it's the ratio between the curve length and the boundary perimeter.
        :type sampling_ratio: float
        """
        super(arc, self).__init__(bc_type, bc_value, order, ref_loc, check_consistency, on_boundary, sampling_ratio)
        self.center = utils.get_complex_input(center)
        self.radius = torch.tensor(radius)
        self.theta1 = torch.tensor(theta1)
        self.theta2 = torch.tensor(theta2)
        self.length = self.radius * torch.abs(self.theta2-self.theta1)

        self.P1 = self.center + self.radius * torch.exp(1.j * self.theta1)
        self.P2 = self.center + self.radius * torch.exp(1.j * self.theta2)
        self.square = [torch.minimum(self.P1.real, self.P2.real), torch.maximum(self.P1.real, self.P2.real), torch.minimum(self.P1.imag, self.P2.imag), torch.maximum(self.P1.imag, self.P2.imag)]
        for i in range(9):
            angle = torch.tensor(-2*torch.pi + i*torch.pi/2)
            if (angle > torch.minimum(self.theta1,self.theta2) and angle < torch.maximum(self.theta1,self.theta2)):
                self.square[0] = torch.minimum(self.square[0], self.center.real + self.radius*torch.cos(angle))
                self.square[1] = torch.maximum(self.square[1], self.center.real + self.radius*torch.cos(angle))
                self.square[2] = torch.minimum(self.square[2], self.center.imag + self.radius*torch.sin(angle))
                self.square[3] = torch.maximum(self.square[3], self.center.imag + self.radius*torch.sin(angle))


    def map_point(self, s): # s in [0,1]
        point = self.center + self.radius * torch.exp(1.j * (self.theta1 + s * (self.theta2 - self.theta1)))
        normal = self.orientation * (point - self.center) / self.radius
        return point, normal
    

    def is_inside(self, points, tol=1e-6):
        dir = points - self.center
        radius = torch.abs(dir)
        angle = torch.angle(dir)
        s = (angle-self.theta1)/(self.theta2-self.theta1) # By definition, this should be in [0,1] (see map_point)
        sp = (angle+2*torch.pi-self.theta1)/(self.theta2-self.theta1) # We also add +-2pi to be sure
        sm = (angle-2*torch.pi-self.theta1)/(self.theta2-self.theta1)
        condition_1 = torch.abs(radius-self.radius) <= tol # The point is on the circle
        condition_2 = ((s>-tol) & (s<1+tol)) | ((sp>-tol) & (sp<1+tol)) | ((sm>-tol) & (sm<1+tol))
        return condition_1 & condition_2


    def intersect_ray(self, points):
        # The problem is that arcs on the left and arcs on the right of the circle define different conditions.
        # So, we split the total arc in left and right sub-arcs and we apply the condition separately.
        # See https://stackoverflow.com/questions/48055362/looking-for-algorithm-that-will-determine-if-a-point-is-to-the-left-right-of-an
        output = torch.zeros(points.shape, dtype=torch.int8)
        tm = torch.minimum(self.theta1, self.theta2)
        tM = torch.maximum(self.theta1, self.theta2)

        arcs = [tm] # Each pair of values in this list are the two angles corresponding to the sub-arc.
        isRight = [torch.cos(tm)>1e-6 or torch.sin(tm)<-1+1e-6] # I.e., when the angle is on the right with the exception of 3pi/2.
        for j in range(9):
            theta = torch.tensor(-4*torch.pi + torch.pi/2 + j*torch.pi) # I.e., -pi/2 and pi/2 for some cycles.
            if (theta > arcs[-1] and theta < tM): # +-pi/2 belongs to the arc, so we have to split the arc into 2 sub-arcs.
                arcs += [theta, theta]
                isRight.append(torch.logical_not(isRight[-1]))
        arcs.append(tM)

        if(torch.abs(tM-tm-2*torch.pi)<1e-10): # First and last sub-arcs might be blended together.
            arcs[-1] = arcs[1] + 2*torch.pi
            arcs.pop(0)
            arcs.pop(0)
            isRight.pop(0)

        for j in range(len(isRight)):
            e1 = self.center + self.radius * torch.exp(1.j * arcs[2*j])
            e2 = self.center + self.radius * torch.exp(1.j * arcs[2*j+1])
            cond1 = torch.logical_and(points.imag < torch.maximum(e1.imag, e2.imag) + 1e-10, points.imag > torch.minimum(e1.imag, e2.imag) - 1e-10)
            if(isRight[j]):
                cond2 = points.real < self.center.real + torch.sqrt(self.radius*self.radius - (points.imag-self.center.imag)*(points.imag-self.center.imag)) + 1e-10
            else:
                cond2 = points.real < self.center.real - torch.sqrt(self.radius*self.radius - (points.imag-self.center.imag)*(points.imag-self.center.imag)) + 1e-10
            
            output[torch.logical_and(cond1, cond2)] += 1

        return torch.remainder(output, 2)



class circle(arc):
    """
    Circle defined as:

    .. math::
        \gamma:=\{z \in \mathbb{C}: z= c + re^{i\\theta}, \\theta \in [0,2\pi)\},

    where :math:`c\in\mathbb{C}` is the center of the circle and :math:`r\in\mathbb{R}` the radius.

    """
    def __init__(self, center, radius, bc_type, bc_value=0, order=1, ref_loc="center", check_consistency=True, on_boundary=True, sampling_ratio=None):
        """
        :param center: Center of the circle.
        :type center: complex / list with size 2 / :class:`torch.tensor` with size 2
        :param radius: Radius of the circle.
        :type radius: float
        :param bc_type: Type of boundary condition to assign to the curve (see above).
        :type bc_type: str
        :param bc_value: Values assigned at the boundary condition. It can be either a constant or a variable value defined through a function.
        :type bc_value: callable / complex
        :param order: Order of sampling refinement at the edges, 1 for no refinement (uniform distribution). For example, if the curve is :math:`[0,1]` and we want to refine on the right side, then the sampled points are distributed as :math:`u^{1/order}`, where :math:`u \sim \mathcal{U}(0,1)` and :math:`order\ge1`.
        :type order: float
        :param ref_loc: Where to apply refinement. 'center' for symmetric refinement (i.e., both edges), 'start' only at the first edge and 'end' only at the second edge.
        :type ref_loc: str
        :param check_consistency: If False, the current curve is excluded from the :func:`pihnn.geometries.boundary.check_consistency` check.
        :type check_consistency: bool
        :param on_boundary: Set to False if the current curve is not really part of the domain boundary.
        :type on_boundary: bool
        :param sampling_ratio: Number of sampled points on the curve with respect to the total sampled points on the boundary. By default, it's the ratio between the curve length and the boundary perimeter.
        :type sampling_ratio: float
        """
        super(circle, self).__init__(center, radius, 0, 2*torch.pi, bc_type, bc_value, order, ref_loc, check_consistency, on_boundary, sampling_ratio)



class crack_tip():
    """
    Object to denote crack tips. Used for enriched networks.

    :param coords: Coordinates of the crack tip.
    :type coords: int/float/complex/list/tuple/tensor
    :param norm: Normal outward vector to the tip.
    :type norm: int/float/complex/list/tuple/tensor
    :param branch_cut_rotation: Rotation of the square root branch cuts with respect to the domain on the left and on the right of the crack, respectively. In radiants and anti-clockwise. Only used in Williams enrichment.
    :type branch_cut_rotation: list of float
    :param initial_sif: Initial values for the trainable stress intensity factors :math:`K_I` and :math:`K_{II}`. Only used in Williams enrichment.
    :type initial_sif: int/float/complex/list/tuple/tensor
    """
    def __init__(self, coords, norm, branch_cut_rotation, initial_sif):
        self.coords = utils.get_complex_input(coords)
        self.norm = utils.get_complex_input(norm)
        self.angle = torch.angle(norm)
        self.branch_cut_rotation = torch.tensor(branch_cut_rotation, device=nn.device).unsqueeze(1)
        self.initial_sif = utils.get_complex_input(initial_sif)