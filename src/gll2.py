import numpy as np


class GLL2:
    """
    stateless class with quadrature points, weights, and basis polynomials
    for linear GLL elements (with 2 points).
    """

    @staticmethod
    def qp():
        """
        return array of quadrature points in reference element [-1,1]
        """
        return np.array([-1, 1])

    @staticmethod
    def qw():
        """
        return array of quadrature weights that correspond to quadrature points from qp()
        """
        return np.array([1, 1])

    @staticmethod
    def phi0(s):
        """
        phi<i>(s) is the basis function associated with qp[i]; phi<i>(s) = 1 if s = qp[i]
        and phi<i>(s) = 0 if s = qp[j] for i != j.
        """
        return 0.5 * (1 - s)

    @staticmethod
    def phi1(s):
        """
        phi<i>(s) is the basis function associated with qp[i]; phi<i>(s) = 1 if s = qp[i]
        and phi<i>(s) = 0 if s = qp[j] for i != j.
        """
        return 0.5 * (1 + s)

    @staticmethod
    def gll_basis():
        return [GLL2.phi0, GLL2.phi1]
