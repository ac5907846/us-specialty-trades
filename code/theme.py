"""Matplotlib style shared by every figure.

Fixed colour roles: cobalt = electrical, green = plumbing/HVAC, amber =
third series (always direct-labelled), rose = LISA hot spots, sienna =
decline. Greys carry structure, never data.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PETROL = "#1b56ad"
OLIVE = "#0e7a43"
ROSE = "#a92458"
SIENNA = "#8a5106"
CLAY = "#d98e12"
CAT = [PETROL, OLIVE, CLAY]

PETROL_TINTS = ["#bdd2ec", "#6f9bd6", "#1b56ad", "#0f3a78"]
OLIVE_TINTS = ["#bfe0cc", "#67b389", "#0e7a43", "#08542d"]
ROSE_TINTS = ["#eec3d4", "#d4749d", "#a92458", "#7a1a40"]

INK = "#1c1e26"
INK_2 = "#494e5c"
INK_3 = "#818697"
RULE = "#c9cdd8"
GRID = "#e9ebf1"
WASH = "#f3f4f8"
SURFACE = "#ffffff"

SEQ = LinearSegmentedColormap.from_list(
    "cobalt_seq", ["#f4f7fb", "#cfdff2", "#9bbde5", "#5c8ecd", "#1b56ad",
                   "#0d2f63"])
SEQ_OLIVE = LinearSegmentedColormap.from_list(
    "green_seq", ["#f3f9f5", "#cfe7d8", "#98ccab", "#54a878", "#0e7a43",
                  "#07472a"])
DIV = LinearSegmentedColormap.from_list(
    "sienna_green_div", ["#6d3d05", "#b5720a", "#e2c184", "#eef0ec",
                         "#a3d0b6", "#3a9c68", "#0b5c33"])
STEPS = ["#dbe5f3", "#adc4e6", "#7c9fd4", "#4d79bf", "#2456a3", "#12386f"]

GOOD = "#0b6b3a"
WARN = "#b5720a"
BAD = "#a92433"


def apply(context="paper"):
    scale = 1.0 if context == "paper" else 1.25
    mpl.use("Agg")
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,

        "font.family": "DejaVu Sans",
        "font.size": 8.5 * scale,
        "axes.titlesize": 9.0 * scale,
        "axes.labelsize": 8.5 * scale,
        "xtick.labelsize": 8.0 * scale,
        "ytick.labelsize": 8.0 * scale,
        "legend.fontsize": 8.0 * scale,

        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "xtick.color": INK_3,
        "ytick.color": INK_3,
        "axes.titlelocation": "left",
        "axes.titlepad": 7,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.edgecolor": RULE,
        "axes.linewidth": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 0,
        "xtick.major.width": 0.8,

        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,

        "lines.linewidth": 1.9,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=CAT),
    })
    return plt


def label_end(ax, x, y, text, color, dx=0.6, dy=0.0, size=8.0, weight="bold"):
    ax.annotate(text, (x + dx, y + dy), color=color, fontsize=size,
                fontweight=weight, va="center", ha="left",
                annotation_clip=False)


def note(ax, text, xy, size=7.2, color=INK_3, ha="left", va="top"):
    ax.annotate(text, xy, xycoords="axes fraction", fontsize=size,
                color=color, ha=ha, va=va, linespacing=1.35)


def panel_tag(ax, letter, title=None, dy=1.06):
    ax.set_title(f"({letter.lower()})", loc="left", color=INK,
                 fontsize=8.5, fontweight="normal", pad=7)


def despine_all(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def rule_line(ax, y=0, color=RULE, lw=0.9, ls="-"):
    ax.axhline(y, color=color, linewidth=lw, linestyle=ls, zorder=0)
