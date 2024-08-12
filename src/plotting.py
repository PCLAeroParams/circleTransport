import numpy as np


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
