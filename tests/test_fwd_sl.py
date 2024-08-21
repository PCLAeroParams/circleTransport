from pathlib import Path
from .context import *

import pytest
import numpy as np


def test_fwd_sl(tmp_path: Path):
    print("========== START TEST: test_fwd_sl ==========")
    d = tmp_path / "plots"
    print(d)
    d.mkdir()
    ne = 6
    dt = 0.1
    div_amp = 0.1
    div_wave_k = 7
    circ0 = CircleNp4(ne)
    print("circ0:", repr(circ0))
    velocity = np.pi / 3 * np.ones_like(circ0.node_theta) + div_amp * np.sin(div_wave_k * circ0.node_theta)
    circ1 = AdvectedCircle(circ0)
    circ1.fwd_euler_step(dt, velocity)
    circ1.check_circle()
    assert np.sum(circ1.elem_arc_len) == pytest.approx(2 * np.pi, rel=1e-15)
    print("circ1:", repr(circ1))

    plotfile = d / "test_plot_2circles.png"
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
