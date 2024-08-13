import numpy as np
import matplotlib.pyplot as plt


def get_color_from_cycle(idx: int):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return colors[idx]


def plot_circle(ax, n=500):
    """
    Plots a high-resolution dashed circle on given plot axes.
    """
    circx = np.cos(np.linspace(-np.pi, np.pi, n))
    circy = np.sin(np.linspace(-np.pi, np.pi, n))
    ax.plot(circx, circy, "k:", label="_nolegend_")
    ax.set_aspect("equal", adjustable="box")


def plot_CircleNp4(ax, circ, cordlabel="_nolegend_", arclabel="_nolegend_"):
    """
    Plots the cord and arc elements of a CircleNp4 instance on a given axes.
    """
    ax.plot(circ.node_cord_x, circ.node_cord_y, ":d", label=cordlabel)
    ax.plot(circ.node_arc_x, circ.node_arc_y, ":s", label=arclabel)
    ax.set_aspect("equal", adjustable="box")


def plot_two_CircleNp4s(ax, circ0, circ1):
    cord_fill = "none"
    circ0_color = get_color_from_cycle(0)
    circ1_color = get_color_from_cycle(1)
    circ0_marker = "d"
    circ1_marker = "o"
    cord_line_style = ":"
    arc_line_style = "-"
    ax.plot(
        circ0.node_cord_x,
        circ0.node_cord_y,
        linestyle=cord_line_style,
        marker=circ0_marker,
        fillstyle=cord_fill,
        color=circ0_color,
        label="circ0.cord",
    )
    ax.plot(
        circ0.node_arc_x, circ0.node_arc_y, linestyle=arc_line_style, marker=circ0_marker, color=circ0_color, label="circ0"
    )
    ax.plot(
        circ1.node_cord_x,
        circ1.node_cord_y,
        linestyle=cord_line_style,
        marker=circ1_marker,
        fillstyle=cord_fill,
        color=circ1_color,
        label="circ1.cord",
    )
    ax.plot(
        circ1.node_arc_x, circ1.node_arc_y, linestyle=arc_line_style, marker=circ1_marker, color=circ1_color, label="circ1"
    )
    ax.set_aspect("equal", adjustable="box")
