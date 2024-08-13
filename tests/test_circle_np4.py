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
    print(repr(circ))
    plot_CircleNp4(ax, circ, "cords", "arcs")
    ax.legend()
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

    fig, (ax0, ax1) = plt.subplots(1, 2, sharex=True, sharey=True)
    #   plot_circle(ax0)
    #   plot_circle(ax1)
    plot_CircleNp4(ax0, circ0, "cords0", "arcs0")
    plot_CircleNp4(ax1, circ1, "cords1", "arcs1")
    ax0.legend()
    ax1.legend()
    plotfile = "test_2plots_2circles.png"
    fig.savefig(plotfile)


def test_overlap(tmp_path: Path):
    pass


#   dt = 0.1
#   circ0 = CircleNp4(ne)
#   print("circ0:", repr(circ0))
#   velocity = np.pi/3 * np.ones_like(circ0.node_theta)
#   circ1 = AdvectedCircle(circ0)
#   circ1.fwd_euler_step(dt, velocity)
#   print("circ1:", repr(circ1))
#
#   d = tmp_path / "plots"
#   d.mkdir()
#   plotfile = "test_plot_2circles.png"
#   fig, ax = plt.subplots()
#   plot_circle(ax)
#   plot_two_CircleNp4s(ax, circ0, circ1)
#   ax.legend()
#   fig.savefig(plotfile)
#
#   circ2 = OverlapNp4(circ0, circ1)
