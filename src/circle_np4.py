import numpy as np


class CircleNp4:
    """
    A unit circle discretization based on the same spectral element type (cubic GLL)
    and reference-to-domain maps as the E3SM atmosphere model.
    """

    @staticmethod
    def reference_to_cord_map(s, x1: float, y1: float, x2: float, y2: float):
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
    def arc_to_reference_map(theta, x0: float, y0: float, x1: float, y1: float):
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
        elem_dth = 2 * np.pi / ne
        self.elems = np.zeros((self.ne, self.np), dtype=int)
        self.elem_cord_len = np.zeros(ne)
        self.elem_arc_len = elem_dth * np.ones(ne)
        # --------------------------
        # mesh construction
        # --------------------------
        self.build_elems_from_boundaries()
        self.reset_theta()

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

    def check_circle(self, test_level=0):
        """
        Property checks.  Returns nerr, where all property tests passing implies
        nerr = 0; for each test that fails, nerr += 1.

        input:
          test_level : if 0, only fast tests will run; if > 0, all tests will run
        """
        fp_tol = 1e-15
        nerr = 0

        #
        # Test circumference of circle = 2*pi
        #
        circumference_rel_err = (self.arc_circumference() - 2 * np.pi) / (2 * np.pi)
        print(f"cord circumference = {self.cord_circumference()}")
        if circumference_rel_err < fp_tol:
            print(f"circumference check (success): arc_circumference = {2*np.pi}")
        else:  # pragma: no cover
            print(
                f"circumference check (ERROR): arc_circumference = {self.arc_circumference()}; rel. err.: {circumference_rel_err}"
            )
            nerr += 1

        #
        # Test that element boundaries match their nodes
        #
        theta_diffs = np.zeros(2)
        for k in range(self.ne):
            theta_diffs[0] = abs(self.elem_theta[k] - self.node_theta[self.elems[k, 0]])
            theta_diffs[1] = abs(self.elem_theta[k + 1] - self.node_theta[self.elems[k, 3]])
            if np.sum(theta_diffs) > fp_tol:  # pragma: no cover
                nerr += 1
        return nerr

    def theta_vals_in_element(self, theta_left, theta_right):
        """
        returns the index of the element that contains a theta value
        """
        left_in = -1
        right_in = -1
        for k in range(self.ne):
            if theta_left >= self.elem_theta[k] and theta_left < self.elem_theta[k + 1]:
                left_in = k
            if theta_right > self.elem_theta[k] and theta_right <= self.elem_theta[k + 1]:
                right_in = k
            if left_in >= 0 and right_in >= 0:
                break
        assert left_in >= 0  # pragma: no cover
        assert right_in >= 0  # pragma: no cover
        return (left_in, right_in)

    def reset_theta(self):
        """
        Reset theta coordinates of nodes and elements to lie within [-pi, pi]
        """
        self.node_theta = np.atan2(self.node_arc_y, self.node_arc_x)
        self.elem_theta = np.atan2(self.elem_y, self.elem_x)

    def update_elem_theta_from_node_theta(self):
        """
        Reset elem theta values to match node theta values at boundaries.
        """
        for k in range(self.ne):
            self.elem_theta[k] = self.node_theta[self.elems[k, 0]]
        self.elem_theta[-1] = self.node_theta[-1]
        for k in range(self.ne):
            self.elem_arc_len[k] = self.elem_theta[k + 1] - self.elem_theta[k]
            dx2 = (np.cos(self.elem_theta[k + 1]) - np.cos(self.elem_theta[k])) ** 2
            dy2 = (np.sin(self.elem_theta[k + 1]) - np.sin(self.elem_theta[k])) ** 2
            self.elem_cord_len[k] = np.sqrt(dx2 + dy2)

    def build_elems_from_boundaries(self):
        """
        Mimics an isoparametric mapping on the sphere.   Element boundaries are
        defined or advected outside of this function, then this function
        builds element interiors based on those boundaries.

        This function will does not use theta values so that it can work with
        advected meshes whose theta values may move outside [-pi, pi]

        """
        node_j = 0  # node index
        for k in range(self.ne):
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

    def __repr__(self):
        nstr = f"CircleNp4: ne = {self.ne}, np = {self.np}\n"
        qstr = f"   qp = {self.qp}\n   qw = {self.qw}\n"
        result = nstr + qstr
        elem_str = "   elem_theta: " + repr(self.elem_theta) + "\n"
        node_str = "   node_theta: " + repr(self.node_theta) + "\n"
        elem_ids = "   elems : " + repr(self.elems) + "\n"
        result += elem_str + node_str + elem_ids
        return result
