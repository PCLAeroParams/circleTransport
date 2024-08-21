from .context import *

import pytest
import numpy as np


def test_gll_qw():
    print("========== START TEST: test_gll_qw ==========")
    gll2 = GLL2()
    assert np.sum(gll2.qw()) == pytest.approx(2, rel=1e-15)

    gll3 = GLL3()
    assert np.sum(gll3.qw()) == pytest.approx(2, rel=1e-15)

    gll4 = GLL3()
    assert np.sum(gll4.qw()) == pytest.approx(2, rel=1e-15)

    gll12 = GLL12()
    assert np.sum(gll12.qw()) == pytest.approx(2, rel=1e-15)


def test_gll_qps():
    print("========== START TEST: test_gll_qps ==========")
    glls = [GLL2(), GLL3(), GLL4()]
    for gll in glls:
        for i, phi in enumerate(gll.gll_basis()):
            for j, qp in enumerate(gll.qp()):
                if i == j:
                    assert phi(qp) == pytest.approx(1, rel=1e-15)
                else:
                    assert phi(qp) == pytest.approx(0, abs=1e-16)
