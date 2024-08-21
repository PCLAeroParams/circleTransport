from .circle_np4 import CircleNp4
from .fwd_sl import AdvectedCircle
import numpy as np


class OverlapNp4:
    """
    An overlap mesh, or 'common refinement' between two different CircleNp4 meshes.
    """

    def check_positive_arc_length(self):
        """
        All arc lengths must be positive; this counts the number of elements
        that are not, and returns that number of errors.
        """
        nerr = 0
        for k in range(self.ne):
            if self.elem_arc_len[k] <= 0:  # pragma: no cover
                print(f"OverlapNp4::check_positive_arc_length ERROR at element {k}")
                nerr += 1
        return nerr

    def circumference(self):
        """
        Returns the circumference, computed as the sum of arc lengths.
        """
        return np.sum(self.elem_arc_len)

    def check_circumference(self):
        """
        Checks that the circumference matchs 2*pi, within floating point roundoff
        """
        nerr = 0
        fp_tol = 1e-14
        if abs(self.circumference() - 2 * np.pi) > fp_tol:  # pragma: no cover
            nerr += 1
            print(f"OverlapNp4::check_circumference ERROR {self.circumference()} != {2*np.pi}")
        return nerr

    def __repr__(self):
        result = "OverlapNp4: \n"
        elem0_str = "   elem0: " + repr(self.elem0) + "\n"
        elem1_str = "   elem1: " + repr(self.elem1) + "\n"
        theta_a_str = "   theta_a: " + repr(np.asarray(self.theta_a)) + "\n"
        theta_b_str = "   theta_b: " + repr(np.asarray(self.theta_b)) + "\n"
        elem_theta_str = "  elem_theta: " + repr(np.asarray(self.elem_theta)) + "\n"
        elem_dtheta_str = "  elem_arc_len: " + repr(np.asarray(self.elem_arc_len)) + "\n"
        len_src_str = "  len_src: " + repr(self.len_src) + "\n"
        len_tgt_str = "  len_tgt: " + repr(self.len_tgt) + "\n"
        result += (
            elem0_str + elem1_str + theta_a_str + theta_b_str + elem_theta_str + elem_dtheta_str + len_tgt_str + len_src_str
        )
        return result

    def __init__(self, circ0: CircleNp4, circ1: AdvectedCircle, verbose: bool = False):
        self.elem0 = []  # self.elem0[k] gives the index, in circ0, that corresponds to overlap element k
        self.elem1 = []  # self.elem1[k] gives the index, in circ1, that corresponds to overlap element k
        self.elem_arc_len = []  # placeholder; will be overwritten
        self.node_theta = []  # placeholder; will be overwritten
        self.elems = []  # placeholder; will be overwritten
        self.elem_x = []  # placeholder; will be overwritten
        self.elem_y = []  # placeholder; will be overwritten
        self.theta_a = []  # left endpoints of overlap elements
        self.theta_b = []  # right endpoints of overlap elements
        self.node_cord_x = []  # placeholder; will be overwritten
        self.node_cord_y = []  # placeholder; will be overwritten
        self.node_arc_x = []  # placeholder; will be overwritten
        self.node_arc_y = []  # placeholder; will be overwritten
        self.mass_matrices = []  # placeholder; will be overwritten
        self.ne = 0
        self.np = circ0.np
        self.qp = circ0.qp.copy()
        self.qw = circ0.qw.copy()
        self.high_order_gll = circ0.high_order_gll
        self.build_overlap(circ0, circ1, verbose)
        self.map_to_reference_coordinates(circ0, circ1)

    def build_overlap(self, circ0: CircleNp4, circ1: AdvectedCircle, verbose: bool = False):
        #
        # Loop over circ1 elements
        #
        # For each circ1 element:
        #    Find circ0 elements that contain circ1 element endpoints, record these indices.
        #    Boundary crossings are flagged with a negative index
        #
        #  Build an overlap list so that overlap element k2 corresponds to elem0[k2] in circ0
        #  and elem1[k2] in circ1; left theta values are self.theta_a[k2] and right theta values are
        #  self.theta_b[k2].
        #
        ov_idx = 0  # index of current overlap element
        for k1 in range(circ1.ne):
            # left endpoint of moved element
            theta_left = circ1.elem_theta[k1]
            # right endpoint of moved element
            theta_right = circ1.elem_theta[k1 + 1]

            if theta_left >= -np.pi and theta_right <= np.pi:
                #
                # case 1: moved element does not encounter periodic domain boundaries
                #
                (left_in, right_in) = circ0.theta_vals_in_element(theta_left, theta_right)

                circ0_index_span = range(left_in, right_in + 1)
                for k0 in circ0_index_span:
                    self.elem0.append(k0)
                    self.elem1.append(k1)
                    self.theta_a.append(max(theta_left, circ0.node_theta[circ0.elems[k0, 0]]))
                    self.theta_b.append(min(theta_right, circ0.node_theta[circ0.elems[k0, 3]]))
                    if verbose:
                        print(
                            f"case 1: overlap elem {ov_idx} is adv. elem {k1} / static elem {k0} theta_left {theta_left} theta_a {self.theta_a[-1]} left_in {left_in} theta_right {theta_right} theta_b {self.theta_b[-1]} right_in {right_in}"
                        )
                    assert self.theta_b[-1] - self.theta_a[-1] > 0
                    ov_idx += 1

            elif theta_left < -np.pi and theta_right >= -np.pi:
                #
                # case 2: moved element straddles left boundary (-pi)
                #
                theta_left += 2 * np.pi
                (left_in, right_in) = circ0.theta_vals_in_element(theta_left, theta_right)
                theta_left -= 2 * np.pi

                circ0_index_span = []
                circ0_bndry_flag = []
                for k0 in range(left_in, circ0.ne):
                    circ0_bndry_flag.append(True)
                    circ0_index_span.append(k0)
                count2 = right_in + 1
                for k0 in range(0, count2):
                    circ0_index_span.append(k0)
                    circ0_bndry_flag.append(False)

                for i, k0 in enumerate(circ0_index_span):
                    self.elem0.append(k0)
                    self.elem1.append(k1)
                    if circ0_bndry_flag[i]:
                        self.theta_a.append(max(theta_left, circ0.node_theta[circ0.elems[k0, 0]] - 2 * np.pi))
                        self.theta_b.append(circ0.node_theta[circ0.elems[k0, 3]] - 2 * np.pi)
                    else:
                        self.theta_a.append(circ0.node_theta[circ0.elems[k0, 0]])
                        self.theta_b.append(min(theta_right, circ0.node_theta[circ0.elems[k0, 3]]))
                    if verbose:
                        print(
                            f"case 2: overlap elem {ov_idx} is adv. elem {k1} / static elem {k0} theta_left {theta_left} theta_a {self.theta_a[-1]} left_in {left_in} theta_right {theta_right} theta_b {self.theta_b[-1]} right_in {right_in}"
                        )
                    assert self.theta_b[-1] - self.theta_a[-1] > 0
                    ov_idx += 1

            elif theta_left <= np.pi and theta_right > np.pi:
                #
                # case 3: moved element straddles right boundary (pi)
                #
                theta_right -= 2 * np.pi
                (left_in, right_in) = circ0.theta_vals_in_element(theta_left, theta_right)
                theta_right += 2 * np.pi

                circ0_index_span = []
                circ0_bndry_flag = []
                for k0 in range(left_in, circ0.ne):
                    circ0_index_span.append(k0)
                    circ0_bndry_flag.append(False)
                for k0 in range(0, right_in + 1):
                    circ0_index_span.append(k0)
                    circ0_bndry_flag.append(True)

                for i, k0 in enumerate(circ0_index_span):
                    self.elem0.append(k0)
                    self.elem1.append(k1)
                    if circ0_bndry_flag[i]:
                        self.theta_a.append(circ0.node_theta[circ0.elems[k0, 0]] + 2 * np.pi)
                        self.theta_b.append(min(theta_right, circ0.node_theta[circ0.elems[k0, 3]] + 2 * np.pi))
                    else:
                        self.theta_a.append(max(theta_left, circ0.node_theta[circ0.elems[k0, 0]]))
                        self.theta_b.append(circ0.node_theta[circ0.elems[k0, 3]])
                    if verbose:
                        print(
                            f"case 3: overlap elem {ov_idx} is adv. elem {k1} / static elem {k0} theta_left {theta_left} theta_a {self.theta_a[-1]} left_in {left_in} theta_right {theta_right} theta_b {self.theta_b[-1]} right_in {right_in}"
                        )
                    assert self.theta_b[-1] - self.theta_a[-1] > 0
                    ov_idx += 1

            elif theta_left < -np.pi and theta_right <= -np.pi:
                #
                # case 4: whole element crossed left boundary (-pi)
                #
                theta_left += 2 * np.pi
                theta_right += 2 * np.pi
                (left_in, right_in) = circ0.theta_vals_in_element(theta_left, theta_right)
                theta_left -= 2 * np.pi
                theta_right -= 2 * np.pi

                circ0_index_span = range(left_in, right_in + 1)
                for k0 in circ0_index_span:
                    self.elem0.append(k0)
                    self.elem1.append(k1)
                    self.theta_a.append(max(theta_left, circ0.node_theta[circ0.elems[k0, 0]] - 2 * np.pi))
                    self.theta_b.append(min(theta_right, circ0.node_theta[circ0.elems[k0, 3]] - 2 * np.pi))
                    if verbose:
                        print(
                            f"case 4: overlap elem {ov_idx} is adv. elem {k1} / static elem {k0} theta_left {theta_left} theta_a {self.theta_a[-1]} left_in {left_in} theta_right {theta_right} theta_b {self.theta_b[-1]} right_in {right_in}"
                        )
                    assert self.theta_b[-1] - self.theta_a[-1] > 0
                    ov_idx += 1

            elif theta_left >= np.pi and theta_right > np.pi:
                #
                # case 5: moved element crossed right boundary (pi)
                #
                theta_left -= 2 * np.pi
                theta_right -= 2 * np.pi
                (left_in, right_in) = circ0.theta_vals_in_element(theta_left, theta_right)
                theta_left += 2 * np.pi
                theta_right += 2 * np.pi

                circ0_index_span = range(left_in, right_in + 1)
                for k0 in circ0_index_span:
                    self.elem0.append(k0)
                    self.elem1.append(k1)
                    self.theta_a.append(max(theta_left, circ0.node_theta[circ0.elems[k0, 0]] + 2 * np.pi))
                    self.theta_b.append(min(theta_right, circ0.node_theta[circ0.elems[k0, 3]] + 2 * np.pi))
                    if verbose:
                        print(
                            f"case 5: overlap elem {ov_idx} is adv. elem {k1} / static elem {k0} theta_left {theta_left} theta_a {self.theta_a[-1]} left_in {left_in} theta_right {theta_right} theta_b {self.theta_b[-1]} right_in {right_in}"
                        )
                    assert self.theta_b[-1] - self.theta_a[-1] > 0
                    ov_idx += 1
            else:  # pragma: no cover
                raise RuntimeError("ERROR: unanticipated case.")
        #
        # Following the above loop, all overlap element arrays should be the same length
        #
        assert len(self.theta_a) == len(self.theta_b)
        assert len(self.theta_a) == len(self.elem0)
        assert len(self.theta_a) == len(self.elem1)
        assert len(self.theta_a) == ov_idx

        #
        # set element boundaries in theta coordinates
        #
        self.ne = len(self.theta_a)
        self.elem_theta = np.zeros(self.ne + 1)
        for k2 in range(0, self.ne, 2):
            self.elem_theta[k2] = self.theta_a[k2]
            self.elem_theta[k2 + 1] = self.theta_b[k2]
        self.elem_theta[-1] = self.theta_b[-1]
        #
        # set arc lengths using moved theta values
        #
        for k2 in range(self.ne):
            self.elem_arc_len.append(self.elem_theta[k2 + 1] - self.elem_theta[k2])
        #
        # define overlap elements in (x,y) space
        #
        self.elem_x = np.cos(self.elem_theta)
        self.elem_y = np.sin(self.elem_theta)

        #
        # reset theta with atan2
        #
        self.elem_theta = np.atan2(self.elem_y, self.elem_x)
        #
        # Check that circumference = 2pi and that all arc lengths are positive
        #
        nerr = self.check_positive_arc_length()
        nerr += self.check_circumference()
        assert nerr == 0

        self.elem_arc_len = np.asarray(self.elem_arc_len)
        self.theta_a = np.asarray(self.theta_a)
        self.theta_b = np.asarray(self.theta_b)

        #
        # build elements
        #
        self.elem_cord_len = np.zeros(self.ne)
        node_j = 0
        self.nn = (self.np - 1) * self.ne + 1
        self.node_cord_x = np.zeros(self.nn)
        self.node_cord_y = np.zeros(self.nn)
        self.node_arc_x = np.zeros(self.nn)
        self.node_arc_y = np.zeros(self.nn)
        self.node_arc_jac = np.zeros(self.nn)
        self.elems = np.zeros((self.ne, self.np), dtype=int)
        self.len_tgt = np.zeros(self.ne)
        self.len_src = np.zeros(self.ne)
        self.elem_x_tgt = np.zeros((self.ne, 2))
        self.elem_y_tgt = np.zeros((self.ne, 2))
        self.elem_x_src = np.zeros((self.ne, 2))
        self.elem_y_src = np.zeros((self.ne, 2))
        for k2 in range(self.ne):
            node_range_start = node_j
            node_range_end = node_j + self.np
            (x1, y1) = (self.elem_x[k2], self.elem_y[k2])
            (x2, y2) = (self.elem_x[k2 + 1], self.elem_y[k2 + 1])
            (cordx, cordy) = CircleNp4.reference_to_cord_map(self.qp, x1, y1, x2, y2)
            radii = np.sqrt(np.square(cordx) + np.square(cordy))
            self.node_arc_x[node_range_start:node_range_end] = cordx / radii
            self.node_arc_y[node_range_start:node_range_end] = cordy / radii
            self.elems[k2, :] = range(node_range_start, node_range_end)
            node_j += self.np - 1

    def check_overlap(self):
        nerr = 0
        circumference = np.sum(self.elem_arc_len)
        fp_tol = 1e-15
        rel_circ_err = abs(circumference - 2 * np.pi) / (2 * np.pi)
        if rel_circ_err > fp_tol:  # pragma: no cover
            nerr += 1
        return nerr

    def map_to_reference_coordinates(self, circ0: CircleNp4, circ1: AdvectedCircle):
        """
        Determines the reference coordinates of each overlap element relative to both
        input meshes.
        """
        for kl in range(self.ne):
            k = self.elem0[kl]  # index of remap target element in circ0
            ll = self.elem1[kl]  # index of remap source element in circ1
            ov_theta_left = self.theta_a[kl]
            ov_theta_right = self.theta_b[kl]
            #
            # find endpoints of overlap element kl in reference space relative to target
            # circ0 element k0
            #
            xy0_tgt = circ0.get_node_xy(k, 0)
            xy1_tgt = circ0.get_node_xy(k, 3)
            tgt_s0 = CircleNp4.theta_to_reference_map(ov_theta_left, xy0_tgt[0], xy0_tgt[1], xy1_tgt[0], xy1_tgt[1])
            tgt_s1 = CircleNp4.theta_to_reference_map(ov_theta_right, xy0_tgt[0], xy0_tgt[1], xy1_tgt[0], xy1_tgt[1])
            (klx0, kly0) = CircleNp4.reference_to_cord_map(tgt_s0, xy0_tgt[0], xy0_tgt[1], xy1_tgt[0], xy1_tgt[1])
            (klx1, kly1) = CircleNp4.reference_to_cord_map(tgt_s1, xy0_tgt[0], xy0_tgt[1], xy1_tgt[0], xy1_tgt[1])
            self.len_tgt[kl] = np.sqrt((klx1 - klx0) ** 2 + (kly1 - kly0) ** 2)
            self.elem_x_tgt[kl, :] = (klx0, klx1)
            self.elem_y_tgt[kl, :] = (kly0, kly1)

            #
            # find endpoints of overlap element kl in reference space relative to source
            # circ1 element l1
            #
            xy0_src = circ1.get_node_xy(ll, 0)
            xy1_src = circ1.get_node_xy(ll, 3)
            src_s0 = CircleNp4.theta_to_reference_map(ov_theta_left, xy0_src[0], xy0_src[1], xy1_src[0], xy1_src[1])
            src_s1 = CircleNp4.theta_to_reference_map(ov_theta_right, xy0_src[0], xy0_src[1], xy1_src[0], xy1_src[1])
            (klx0, kly0) = CircleNp4.reference_to_cord_map(src_s0, xy0_src[0], xy0_src[1], xy1_src[0], xy1_src[1])
            (klx1, kly1) = CircleNp4.reference_to_cord_map(src_s1, xy0_src[0], xy0_src[1], xy1_src[0], xy1_src[1])
            self.len_src[kl] = np.sqrt((klx1 - klx0) ** 2 + (kly1 - kly0) ** 2)
            self.elem_x_src[kl, :] = (klx0, klx1)
            self.elem_y_src[kl, :] = (kly0, kly1)
