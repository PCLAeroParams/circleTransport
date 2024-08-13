from . import CircleNp4
import numpy as np


class AdvectedCircle(CircleNp4):
    """
    We require a second class for an advected mesh because python won't deepcopy class
    instances (even with the deepcopy method from the copy module).
    """

    def __init__(self, other: CircleNp4):
        """
        Constructor. Deep copies an existing CircleNp4
        """
        self.np = other.np
        self.ne = other.ne
        self.qp = other.qp.copy()
        self.qw = other.qw.copy()
        self.nn = other.nn
        self.node_cord_x = other.node_cord_x.copy()
        self.node_cord_y = other.node_cord_y.copy()
        self.node_arc_x = other.node_arc_x.copy()
        self.node_arc_y = other.node_arc_y.copy()
        self.node_theta = other.node_theta.copy()
        self.node_arc_jac = other.node_arc_jac.copy()
        self.elem_theta = other.elem_theta.copy()
        self.elem_x = other.elem_x.copy()
        self.elem_y = other.elem_y.copy()
        self.elems = other.elems.copy()
        self.elem_cord_len = other.elem_cord_len.copy()
        self.elem_arc_len = other.elem_arc_len.copy()

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
        self.node_theta += dt * node_theta_velocity
        self.update_elem_theta_from_node_theta()
        self.elem_x = np.cos(self.elem_theta)
        self.elem_y = np.sin(self.elem_theta)
        self.build_elems_from_boundaries()
