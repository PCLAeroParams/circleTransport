import numpy as np


class GLL4:
    """
    stateless class with quadrature points, weights, and basis polynomials
    for cubic GLL elements (with 4 points).
    """

    @staticmethod
    def qp():
        """
        return array of quadrature points in reference element [-1,1]
        """
        return np.array([-1, -1 / np.sqrt(5), 1 / np.sqrt(5), 1])

    @staticmethod
    def qw():
        """
        return array of quadrature weights that correspond to quadrature points from qp()
        """
        return np.array([1, 5, 5, 1]) / 6.0

    @staticmethod
    def phi0(s):
        """
        phi<i>(s) is the basis function associated with qp[i]; phi<i>(s) = 1 if s = qp[i]
        and phi<i>(s) = 0 if s = qp[j] for i != j.
        """
        return (-1 + s + 5 * s**2 - 5 * s**3) / 8

    @staticmethod
    def phi1(s):
        """
        phi<i>(s) is the basis function associated with qp[i]; phi<i>(s) = 1 if s = qp[i]
        and phi<i>(s) = 0 if s = qp[j] for i != j.
        """
        return 5 * (s - 1) * (s + 1) * (np.sqrt(5) * s - 1) / 8

    @staticmethod
    def phi2(s):
        """
        phi<i>(s) is the basis function associated with qp[i]; phi<i>(s) = 1 if s = qp[i]
        and phi<i>(s) = 0 if s = qp[j] for i != j.
        """
        return -np.sqrt(5) * (s - 1) * (s + 1) * (np.sqrt(5) + 5 * s) / 8

    @staticmethod
    def phi3(s):
        """
        phi<i>(s) is the basis function associated with qp[i]; phi<i>(s) = 1 if s = qp[i]
        and phi<i>(s) = 0 if s = qp[j] for i != j.
        """
        return (-1 - s + 5 * s**2 + 5 * s**3) / 8
