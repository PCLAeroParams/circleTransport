from pathlib import Path
from .context import *

import pytest
import numpy as np


def test_circle():
    print("========== START TEST: test_circle ==========")
    nes = [6, 12, 24]
    for ne in nes:
        circ = CircleNp4(ne)
        print(f"circ.gll_circumference() = {circ.gll_circumference()}")
        test_level = 1
        nerr = circ.check_circle(test_level)
        ntheta = 10
        theta_check = np.linspace(-np.pi + np.pi / 20, np.pi - np.pi / 20, ntheta)
        for i in range(ntheta - 1):
            (left_idx, right_idx) = circ.theta_vals_in_element(theta_check[i], theta_check[i + 1])
        if nerr > 0:  # pragma: no cover
            print(f"check_circle found {nerr} errors.")
        assert nerr == 0
        assert circ.arc_circumference() == pytest.approx(2 * np.pi, rel=1e-15)


def test_plot_circle(tmp_path: Path):
    print("========== START TEST: test_plot_circle ==========")
    d = tmp_path / "plots"
    d.mkdir()
    plotfile = "test_plot_circle.png"
    fig, ax = plt.subplots()
    plot_circle(ax)
    circ = CircleNp4(6)
    plot_CircleNp4(ax, circ, "nodes")
    ax.legend()
    fig.savefig(plotfile)

    plotfile = "test_elem_plot.png"
    fig, ax = plt.subplots()
    plot_circle_by_element(ax, circ)
    fig.savefig(plotfile)


def test_maps():
    print("========== START TEST: test_maps  ==========")
    thetak = np.pi / 20
    thetak1 = np.pi / 3
    (xk, yk) = (np.cos(thetak), np.sin(thetak))
    (xk1, yk1) = (np.cos(thetak1), np.sin(thetak1))

    # check that endpoints in r  map to endpoints of element
    (xkcheck, ykcheck) = CircleNp4.reference_to_arc_map(-1, xk, yk, xk1, yk1)
    (xk1check, yk1check) = CircleNp4.reference_to_arc_map(1, xk, yk, xk1, yk1)
    assert xkcheck == pytest.approx(xk, rel=1e-15)
    assert ykcheck == pytest.approx(yk, rel=1e-15)
    assert xk1check == pytest.approx(xk1, rel=1e-15)
    assert yk1check == pytest.approx(yk1, rel=1e-15)

    # check that the cord and arc maps agree
    s = 5.0 / 8
    (xc, yc) = CircleNp4.reference_to_cord_map(s, xk, yk, xk1, yk1)
    rc = np.sqrt(xc**2 + yc**2)
    (xq, yq) = (xc / rc, yc / rc)
    (xq1, yq1) = CircleNp4.reference_to_arc_map(s, xk, yk, xk1, yk1)
    thetaq = np.atan2(yq, xq)
    thetar = CircleNp4.reference_to_theta_map(s, xk, yk, xk1, yk1)
    assert np.square(xq1 - xq) + np.square(yq1 - yq) == pytest.approx(0, rel=1e-15)
    assert np.atan2(yc, xc) == pytest.approx(thetaq, rel=1e-15)
    assert thetaq == pytest.approx(thetar, rel=1e-15)

    # check map inverses
    sk = CircleNp4.theta_to_reference_map(thetak, xk, yk, xk1, yk1)
    sk1 = CircleNp4.theta_to_reference_map(thetak1, xk, yk, xk1, yk1)
    sq = CircleNp4.theta_to_reference_map(thetaq, xk, yk, xk1, yk1)
    assert sq == pytest.approx(s, rel=1e-15)
    assert sk == pytest.approx(-1, rel=1e-15)
    assert sk1 == pytest.approx(1, rel=1e-15)
