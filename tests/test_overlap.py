from pathlib import Path
from .context import *

import pytest
import numpy as np


def test_overlap(tmp_path: Path):
    print("========== START TEST: test_overlap ==========")
    #   dts = [0.1, 0.5, 1, -0.1, -0.3, -0.8]
    dts = [-0.1, 0.1, -1.5, 1.5]
    ne = 6
    circ0 = CircleNp4(ne)
    div_amp = 0.1
    div_wave_k = 7
    velocity = np.pi / 3 * np.ones_like(circ0.node_theta) + div_amp * np.sin(div_wave_k * circ0.node_theta)
    print("circ0:", repr(circ0))
    for i, dt in enumerate(dts):
        circ1 = AdvectedCircle(circ0)
        circ1.fwd_euler_step(dt, velocity)
        nerr = circ1.check_circle()
        assert nerr == 0
        print("circ1:", repr(circ1))

        overlap = OverlapNp4(circ0, circ1)
        nerr = overlap.check_overlap()
        assert nerr == 0
        print("circ2:", repr(overlap))

        plotfile = "test_overlap" + str(i) + ".png"
        fig, (ax0, ax1) = plt.subplots(1, 2, sharex=True, sharey=True)
        plot_two_circles_by_element(ax0, circ0, circ1)
        plot_overlap(ax1, overlap)
        fig.savefig(plotfile)
        plt.close(fig)


def test_self_overlap():
    print("========== START TEST: test_self_overlap ==========")
    ne = 8
    dt = 1
    circ0 = CircleNp4(ne)
    velocity = np.zeros_like(circ0.node_theta)
    circ1 = AdvectedCircle(circ0)
    circ1.fwd_euler_step(dt, velocity)
    assert np.sum(circ1.density_factors) == pytest.approx(ne)
    overlap = OverlapNp4(circ0, circ1)
    print("circ1  : ", repr(circ1))
    print("overlap: ", repr(overlap))
    nerr = overlap.check_overlap()
    assert nerr == 0
