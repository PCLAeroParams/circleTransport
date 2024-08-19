import numpy as np


class GLL3:
    """
    stateless class with quadrature points, weights, and basis polynomials
    for quadratic GLL elements (with 3 points).
    """

    @staticmethod
    def qp():
        """
        return array of quadrature points in reference element [-1,1]
        """
        return np.array([-1, 0, 1])

    @staticmethod
    def qw():
        """
        return array of quadrature weights that correspond to quadrature points from qp()
        """
        return np.array([1, 4, 1]) / 3.0

    @staticmethod
    def phi0(s):
        """
        phi<i>(s) is the basis function associated with qp[i]; phi<i>(s) = 1 if s = qp[i]
        and phi<i>(s) = 0 if s = qp[j] for i != j.
        """
        return 0.5 * s * (s - 1)

    @staticmethod
    def phi1(s):
        """
        phi<i>(s) is the basis function associated with qp[i]; phi<i>(s) = 1 if s = qp[i]
        and phi<i>(s) = 0 if s = qp[j] for i != j.
        """
        return -(s - 1) * (s + 1)

    @staticmethod
    def phi2(s):
        """
        phi<i>(s) is the basis function associated with qp[i]; phi<i>(s) = 1 if s = qp[i]
        and phi<i>(s) = 0 if s = qp[j] for i != j.
        """
        return 0.5 * s * (s + 1)

    @staticmethod
    def gll_basis():
        return [GLL3.phi0, GLL3.phi1, GLL3.phi2]
