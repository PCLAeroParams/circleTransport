from .context import *

import pytest
import numpy as np


def test_circle():
    nes = [6, 12, 24]
    for ne in nes:
        circ = CircleNp4(ne)
        nerr = circ.check_circle()
        assert nerr == 0
        assert circ.arc_circumference() == pytest.approx(2 * np.pi, rel=1e-15)


def test_plot_circle(tmp_path):
    d = tmp_path / "plots"
    d.mkdir()
    plotfile = "test_plot_circle.png"
    fig, ax = plt.subplots()
    plot_circle(ax)
    circ = CircleNp4(6)
    plot_CircleNp4(ax, circ, "cords", "arcs")
    ax.legend()
    fig.savefig(plotfile)
