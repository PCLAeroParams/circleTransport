import numpy as np


class CircleNp4:
    @staticmethod
    def reference_to_cord_map(s, x1, y1, x2, y2):
        """
        Map a set of points from the reference element to a cord

        input:
          s: reference coordinates in [-1,1]
          x1, y1 : x, y coordinates of cord's left endpoint
          x2, y2 : x, y coordinates of cord's right endpoint

        output:
          [cordx, cordy] array of x coordinates, array of y coordinates for mapped points
        """
        cordx = 0.5 * ((1 - s) * x1 + (1 + s) * x2)
        cordy = 0.5 * ((1 - s) * y1 + (1 + s) * y2)
        return (cordx, cordy)

    @staticmethod
    def reference_to_arc_jacobian(s, dtheta):
        """
        Computes the Jacobian determinant of the mapping from the reference element [-1,1]
        to an arc on the unit circle.

        input:
          s : reference element coordinates
          dtheta : arc length of the image on the unit circle
        """
        denom = 1 + np.square(s) + (1 - np.square(s)) * np.cos(dtheta)
        return np.sin(dtheta) / denom

    @staticmethod
    def arc_to_reference_map(theta, x0, y0, x1, y1):
        """
        Given a point on the unit circle, with coordinate theta, contained in the element
        whose endpoints are given, find the corresponding reference element coordinate in [-1,1]

        input:
          theta : theta coordinate for point on unit circle
          x0, y0 : x-y coordinates of the left endpoint of the element that contains theta
          x1, y1 : x-y coordinates of the right endpoint of the element that contains theta

        output:
          s : reference coordinate of theta in [-1,1]
        """
        dy = y1 - y0
        dx = x1 - x0
        m = dy / dx
        ax = np.cos(theta)
        ay = np.sin(theta)
        rth = (y0 - m * x0) / (ay - m * ax)
        x = rth * ax
        y = rth * ay
        norm0sq = x0 * x0 + y0 * y0
        cord_dot = x0 * x1 + y0 * y1
        pts_dot = x0 * x + y0 * y

        s = (2 * pts_dot - norm0sq - cord_dot) / (cord_dot - norm0sq)
        return s

    """
    A unit circle discretization based on the same spectral element type (cubic GLL)
    and reference-to-domain maps as the E3SM atmosphere model.
  """

    def __init__(self, ne):
        # --------------------------
        # global attributes
        # --------------------------
        self.np = 4
        self.ne = ne
        self.qp = np.array([-1, -1 / np.sqrt(5.0), 1 / np.sqrt(5.0), 1])
        self.qw = np.array([1.0 / 6, 5.0 / 6, 5.0 / 6, 1.0 / 6])
        self.nn = (self.np - 1) * ne + 1
        # --------------------------
        # node attributes
        # --------------------------
        self.node_cord_x = np.zeros(self.nn)
        self.node_cord_y = np.zeros(self.nn)
        self.node_arc_x = np.zeros(self.nn)
        self.node_arc_y = np.zeros(self.nn)
        self.node_arc_jac = np.zeros(self.nn)
        # --------------------------
        # element attributes
        # --------------------------
        self.elem_theta = np.linspace(-np.pi, np.pi, ne + 1)
        self.elem_x = np.cos(self.elem_theta)
        self.elem_y = np.sin(self.elem_theta)
        self.elem_dth = self.elem_theta[1] - self.elem_theta[0]
        self.elems = np.zeros((self.ne, self.np), dtype=int)
        self.elem_cord_len = np.zeros(ne)
        self.elem_arc_len = self.elem_dth * np.ones(ne)
        # --------------------------
        # mesh construction
        # --------------------------
        node_j = 0  # node index
        for k in range(ne):
            node_range_start = node_j
            node_range_end = node_j + self.np
            dtheta_k = self.elem_theta[k + 1] - self.elem_theta[k]
            x1 = self.elem_x[k]
            y1 = self.elem_y[k]
            x2 = self.elem_x[k + 1]
            y2 = self.elem_y[k + 1]
            self.elem_cord_len[k] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            (cordx, cordy) = CircleNp4.reference_to_cord_map(self.qp, x1, y1, x2, y2)
            radii = np.sqrt(np.square(cordx) + np.square(cordy))
            self.node_cord_x[node_range_start:node_range_end] = cordx
            self.node_cord_y[node_range_start:node_range_end] = cordy
            self.node_arc_x[node_range_start:node_range_end] = cordx / radii
            self.node_arc_y[node_range_start:node_range_end] = cordy / radii
            self.node_arc_jac[node_range_start:node_range_end] = CircleNp4.reference_to_arc_jacobian(self.qp, dtheta_k)
            self.elems[k, :] = range(node_j, node_j + self.np)
            node_j += self.np - 1
        self.node_theta = np.atan2(self.node_arc_y, self.node_arc_x)

    def cord_circumference(self):
        """
        return the sum of all cord lengths
        """
        return np.sum(self.elem_cord_len)

    def arc_circumference(self):
        """
        return the sum of all arc lengths
        """
        return np.sum(self.elem_arc_len)

    def check_circle(self):
        fp_tol = 1e-15
        nerr = 0
        circumference_rel_err = (self.arc_circumference() - 2 * np.pi) / (2 * np.pi)
        print(f"cord circumference = {self.cord_circumference()}")
        if circumference_rel_err < fp_tol:
            print(f"circumference check (success): arc_circumference = {2*np.pi}")
        else:  # pragma: no cover
            print(
                f"circumference check (ERROR): arc_circumference = {self.arc_circumference()}; rel. err.: {circumference_rel_err}"
            )
            nerr += 1
        return nerr
