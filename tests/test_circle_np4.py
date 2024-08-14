from pathlib import Path
from .context import *

import pytest
import numpy as np


def test_circle():
    nes = [6, 12, 24]
    for ne in nes:
        circ = CircleNp4(ne)
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
    d = tmp_path / "plots"
    d.mkdir()
    plotfile = "test_plot_circle.png"
    fig, ax = plt.subplots()
    plot_circle(ax)
    circ = CircleNp4(6)
    plot_CircleNp4(ax, circ, "cords", "arcs")
    ax.legend()
    fig.savefig(plotfile)

    plotfile = "test_elem_plot.png"
    fig, ax = plt.subplots()
    plot_circle_by_element(ax, circ)
    fig.savefig(plotfile)


def test_fwd_sl(tmp_path: Path):
    d = tmp_path / "plots"
    d.mkdir()
    ne = 6
    dt = 0.1
    circ0 = CircleNp4(ne)
    print("circ0:", repr(circ0))
    velocity = np.pi / 3 * np.ones_like(circ0.node_theta)
    circ1 = AdvectedCircle(circ0)
    circ1.fwd_euler_step(dt, velocity)
    print("circ1:", repr(circ1))

    plotfile = "test_plot_2circles.png"
    fig, ax = plt.subplots()
    plot_circle(ax)
    plot_two_CircleNp4s(ax, circ0, circ1)
    ax.legend()
    fig.savefig(plotfile)
    plt.close(fig)

    plotfile = "test_plot_2elems.png"
    fig, ax = plt.subplots()
    plot_two_circles_by_element(ax, circ0, circ1)
    fig.savefig(plotfile)
    plt.close(fig)


def test_overlap(tmp_path: Path):
    #   dts = [0.1, 0.5, 1, -0.1, -0.3, -0.8]
    dts = [-0.1, 0.1, -1.5, 1.5]
    ne = 6
    circ0 = CircleNp4(ne)
    print("circ0:", repr(circ0))
    for dt in dts:
        velocity = np.pi / 3 * np.ones_like(circ0.node_theta)
        circ1 = AdvectedCircle(circ0)
        circ1.fwd_euler_step(dt, velocity)
        print("circ1:", repr(circ1))

        overlap = OverlapNp4(circ0, circ1)
        print("circ2:", repr(overlap))
