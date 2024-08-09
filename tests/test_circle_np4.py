from ..circle_np4 import CircleNp4

import pytest
import numpy as np


def test_circle():
    nes = [6, 12, 24]
    for ne in nes:
        circ = CircleNp4(ne)
        circ.check_circle()
        assert circ.arc_circumference() == pytest.approx(2 * np.pi, rel=1e-15)
