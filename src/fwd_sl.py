from .circle_np4 import CircleNp4
from .gll12 import GLL12
import numpy as np


class AdvectedCircle(CircleNp4):
    """ """

    def __init__(self, other: CircleNp4):
        """
        Constructor. Deep copies an existing CircleNp4
        """
        self.np = other.np
        self.ne = other.ne
        self.qp = other.qp.copy()
        self.qw = other.qw.copy()
        self.nn = other.nn
        self.high_order_gll = GLL12()
        self.node_arc_x = other.node_arc_x.copy()
        self.node_arc_y = other.node_arc_y.copy()
        self.node_theta = other.node_theta.copy()
        self.elem_theta = other.elem_theta.copy()
        self.elem_x = other.elem_x.copy()
        self.elem_y = other.elem_y.copy()
        self.elems = other.elems.copy()
        self.elem_arc_len = other.elem_arc_len.copy()
        self.mass_matrices = other.mass_matrices.copy()
        self.density_factors = np.ones_like(self.node_arc_x)

    def __repr__(self):
        result = CircleNp4.__repr__(self)
        rho_fac_str = "  density_factors : " + repr(self.density_factors)
        return result + rho_fac_str

    def compute_density_factors(self, prev_arc_len):
        self.density_factors = prev_arc_len / self.elem_arc_len

    def update_elem_theta_from_node_theta(self):
        """
        Reset elem theta values to match node theta values at boundaries.
        """
        for k in range(self.ne):
            self.elem_theta[k] = self.node_theta[self.elems[k, 0]]
        self.elem_theta[-1] = self.node_theta[-1]
        for k in range(self.ne):
            self.elem_arc_len[k] = self.elem_theta[k + 1] - self.elem_theta[k]

        fp_tol = 1e-15
        for k in range(self.ne):
            xye0 = self.get_elem_xy(k)
            xye1 = self.get_elem_xy(k + 1)
            xyn0 = self.get_node_xy(k, 0)
            xyn1 = self.get_node_xy(k, 3)

            assert np.sum(np.square(xyn0 - xye0)) < fp_tol
            assert np.sum(np.square(xyn1 - xye1)) < fp_tol

    def fwd_euler_step(self, dt, node_theta_velocity):
        """
        Advects a copied mesh forward in time (by moving each mesh node)
        one step using Euler's method.

        input:
          dt : time step size
          theta_velocity : [circ.ne] array. velocity at each node of circ's mesh.
              scalars: defined relative to theta coordinate

        output:
          advected mesh
            **coordinates are not reset to [-pi, pi]** by this method
            element boundaries are moved (possibly outside of the [-pi, pi] interval)
            then element interiors are reconstructed

        """
        prev_arc_len = self.elem_arc_len.copy()
        self.node_theta += dt * node_theta_velocity
        self.update_elem_theta_from_node_theta()
        self.elem_x = np.cos(self.elem_theta)
        self.elem_y = np.sin(self.elem_theta)
        self.reset_xy_from_theta()
        self.reset_arc_len()
        self.compute_density_factors(prev_arc_len)


#         print(f"density_factors: {repr(self.density_factors)}")
#         self.build_elems_from_boundaries()
