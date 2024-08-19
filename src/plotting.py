import numpy as np
import matplotlib.pyplot as plt


def get_color_from_cycle(idx: int):
    """
    Get a color from the default property cycle

    input:
      idx: index of color in cycle
    output:
      color (rgb) value
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return colors[idx % len(colors)]


def plot_circle(ax, n=500):
    """
    Plots a high-resolution dashed circle on given plot axes.
    """
    circx = np.cos(np.linspace(-np.pi, np.pi, n))
    circy = np.sin(np.linspace(-np.pi, np.pi, n))
    ax.plot(circx, circy, "k:", label="_nolegend_")
    ax.set_aspect("equal", adjustable="box")


def plot_circle_by_element(ax, circ):
    """
    Plots a CircleNp4 by element
    """
    for k in range(circ.ne):
        elem_color = get_color_from_cycle(k)
        arc_line = "-"
        arc_marker = "s"
        ax.plot(
            circ.node_arc_x[circ.elems[k, :]],
            circ.node_arc_y[circ.elems[k, :]],
            linestyle=arc_line,
            marker=arc_marker,
            color=elem_color,
            label=str(k),
        )
    ax.set_aspect("equal", adjustable="box")
    ax.legend()


def plot_overlap(ax, overlap):
    """
    Plots an overlap mesh on a set of axes
    """
    arc_line = "-"
    arc_marker = "s"
    for k in range(overlap.ne):
        elem_color = get_color_from_cycle(k)
        ax.plot(
            overlap.node_arc_x[overlap.elems[k, :]],
            overlap.node_arc_y[overlap.elems[k, :]],
            linestyle=arc_line,
            marker=arc_marker,
            color=elem_color,
            label=f"k-l = {overlap.elem0[k]}-{overlap.elem1[k]}",
        )
        ax.set_aspect("equal", adjustable="box")
        ax.legend()


def plot_two_circles_by_element(ax, circ0, circ1):
    for k in range(circ0.ne):
        elem_color = get_color_from_cycle(k)
        arc_line = "-"
        arc_marker = "s"
        ax.plot(
            circ0.node_arc_x[circ0.elems[k, :]],
            circ0.node_arc_y[circ0.elems[k, :]],
            linestyle=arc_line,
            marker=arc_marker,
            color=elem_color,
            label=str(k),
        )
    for k in range(circ1.ne):
        elem_color = get_color_from_cycle(k)
        arc_line = "-"
        arc_marker = "s"
        ax.plot(
            circ1.node_arc_x[circ1.elems[k, :]],
            circ1.node_arc_y[circ1.elems[k, :]],
            linestyle=arc_line,
            marker=arc_marker,
            color=elem_color,
            fillstyle="none",
            label="_nolegend_",
        )
    ax.set_aspect("equal", adjustable="box")
    ax.legend()


def plot_CircleNp4(ax, circ, arclabel="_nolegend_"):
    """
    Plots the arc elements of a CircleNp4 instance on a given axes.
    """
    c0 = get_color_from_cycle(0)
    ax.plot(circ.node_arc_x, circ.node_arc_y, ":s", color=c0, label=arclabel)
    ax.set_aspect("equal", adjustable="box")


def plot_two_CircleNp4s(ax, circ0, circ1):
    """
    Plots two circles on the same axes.
    """
    circ0_color = get_color_from_cycle(0)
    circ1_color = get_color_from_cycle(1)
    circ0_marker = "d"
    circ1_marker = "o"
    arc_line_style = "-"

    ax.plot(
        circ0.node_arc_x, circ0.node_arc_y, linestyle=arc_line_style, marker=circ0_marker, color=circ0_color, label="circ0"
    )
    ax.plot(
        circ1.node_arc_x, circ1.node_arc_y, linestyle=arc_line_style, marker=circ1_marker, color=circ1_color, label="circ1"
    )
    ax.set_aspect("equal", adjustable="box")
