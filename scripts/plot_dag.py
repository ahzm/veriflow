#!/usr/bin/env python3
# scripts/plot_dag.py

from __future__ import annotations

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Dict, Any, List, Tuple, Optional
import json
import typer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import networkx as nx
from networkx.exception import NetworkXUnfeasible

from veriflow.utils.graph import build_dag
from veriflow.utils.io import save_fig
from veriflow.utils.logger import log


app = typer.Typer(help="Plot a DAG from an n8n workflow and optionally highlight the executed path.")

# ---------- IO helpers ----------
def _load_workflow(fp: Path) -> Dict[str, Any]:
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_detail(fp: Optional[Path]) -> Dict[str, Any]:
    if fp is None:
        return {}
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_name_map(workflow: Dict[str, Any]) -> Dict[str, str]:
    """Map node id -> name (fallback to id)."""
    name_map: Dict[str, str] = {}
    for n in workflow.get("nodes", []):
        nid = str(n.get("id"))
        if nid and nid != "None":
            name_map[nid] = n.get("name", nid)
    return name_map


def _executed_path_from_detail(detail: Dict[str, Any]) -> List[str]:
    """Prefer executed_nodes_readable; fallback to ids."""
    if not detail:
        return []
    # detail may be whole verify report or only exec_detail
    exec_detail = detail.get("exec_detail", detail)
    if "executed_nodes_readable" in exec_detail:
        return [str(x) for x in exec_detail["executed_nodes_readable"]]
    if "executed_nodes" in exec_detail:
        return [str(x) for x in exec_detail["executed_nodes"]]
    return []

# ---------- layout & drawing ----------
def _to_named_graph(G: nx.DiGraph, name_map: Dict[str, str], use_name: bool) -> nx.DiGraph:
    """Return a name-keyed DiGraph when use_name=True; otherwise keep keys as-is."""
    H = nx.DiGraph()
    if use_name:
        def lbl(x: Any) -> str:
            sx = str(x)
            return name_map.get(sx, sx)
        H.add_nodes_from([lbl(n) for n in G.nodes])
        H.add_edges_from([(lbl(u), lbl(v)) for (u, v) in G.edges])
    else:
        H.add_nodes_from(G.nodes)
        H.add_edges_from(G.edges)
    return H

def _auto_layout(H: nx.DiGraph) -> Dict[Any, Tuple[float, float]]:
    """Select a decent layout for small/medium graphs."""
    try:
        if H.number_of_nodes() <= 10:
            return nx.kamada_kawai_layout(H)
    except Exception:
        pass
    return nx.spring_layout(H, k=0.7, iterations=200, seed=42)

def _draw_rounded_node(ax, xy, text, facecolor, edgecolor="#3c3c3c", shadow=True):
    """Draw a rounded box with optional shadow and centered label."""
    x, y = xy
    w, h = 0.28, 0.14  # box width/height in data coords (works well for small graphs)

    if shadow:
        # simple shadow offset
        shadow_box = FancyBboxPatch(
            (x - w/2 + 0.015, y - h/2 - 0.015), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.04",
            linewidth=0, facecolor="0.80", alpha=0.4, zorder=1
        )
        ax.add_patch(shadow_box)

    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.02",
        linewidth=1.2, edgecolor=edgecolor, facecolor=facecolor, zorder=2
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=10, zorder=3)

def _edge_list_from_path(path: List[str]) -> List[Tuple[str, str]]:
    return list(zip(path, path[1:]))

def _layered_layout(H: nx.DiGraph, horizontal: bool = True,
                    box_w: float = 0.28, box_h: float = 0.14,
                    layer_gap_factor: float = 3.0, row_gap_factor: float = 2.4):
    """
    Layered DAG layout using data-units consistent with node box size.
    layer_gap = layer_gap_factor * box_w
    row_gap   = row_gap_factor   * box_h
    """
    # 1) compute topological levels
    level = {}
    try:
        order = list(nx.topological_sort(H))
        for n in order:
            preds = list(H.predecessors(n))
            level[n] = (max(level[p] for p in preds) + 1) if preds else 0
    except NetworkXUnfeasible:
        print("[warn] Graph contains cycles — falling back to spring layout.")
        return nx.spring_layout(H, k=0.7, iterations=200, seed=42)
    
    
    # 2) group by level, keep deterministic order
    layers: Dict[int, List[Any]] = {}
    for n, L in level.items():
        layers.setdefault(L, []).append(n)
    for L in layers:
        layers[L].sort(key=str)

    # 3) spacing derived from box size
    layer_gap = layer_gap_factor * box_w
    row_gap   = row_gap_factor   * box_h

    # 4) assign positions
    pos: Dict[Any, Tuple[float, float]] = {}
    for L, nodes in layers.items():
        for i, n in enumerate(nodes):
            if horizontal:
                x = L * layer_gap
                y = -(i * row_gap)
            else:
                x = i * row_gap
                y = -(L * layer_gap)
            pos[n] = (x, y)
    return pos

def _edge_margins(ax, box_w, box_h):
    src = data_to_points(box_w * 0.45, box_h * 0.45, ax) + 1.0
    tgt = data_to_points(box_w * 0.48, box_h * 0.48, ax) + 1.0
    return max(6.0, src), max(6.0, tgt)

def data_to_points(dx: float, dy: float, ax) -> float:
    """Convert data-unit delta to points (approx length)."""
    import numpy as np
    # data -> display (pixels)
    x0, y0 = ax.transData.transform((0,0))
    x1, y1 = ax.transData.transform((dx,dy))
    # pixels -> points (1 pt = 1/72 inch ; fig.dpi pixels per inch)
    pix_len = np.hypot(x1 - x0, y1 - y0)
    return 72.0 * pix_len / ax.figure.dpi

# ---------- color/theme ----------
BASE_NODE = "#5B8FD9"    # base blue
PATH_START = "#2AA876"   # start: green
PATH_MID   = "#6CCECB"   # middle: teal (if long)
PATH_END   = "#F4A259"   # end: orange
EDGE_BASE  = "#888888"
EDGE_HI    = "#E4572E"   # highlighted edge color
FONT_FAMILY = "DejaVu Sans"  # widely available

@app.command()
def plot(
    input: Path = typer.Option(..., "--input", "-i", exists=True, readable=True, help="Path to n8n workflow JSON"),
    detail: Optional[Path] = typer.Option(None, "--detail", "-d", help="Path to veriflow detail/report JSON"),
    out: Path = typer.Option(Path("experiments/results/dag.png"), "--out", "-o", help="Output image path"),
    title: Optional[str] = typer.Option(None, "--title", help="Figure title"),
    label_by: str = typer.Option("name", "--label-by", help="name|id (label preference)"),
    show_legend: bool = typer.Option(True, "--legend/--no-legend", help="Show legend"),
):
    """Plot a publication-friendly DAG with rounded nodes and highlighted executed path."""
    matplotlib.rcParams["font.family"] = FONT_FAMILY

    wf = _load_workflow(input)
    det = _load_detail(detail)
    log.info(f"Loaded workflow: {input}")
    if detail:
        log.info(f"Loaded detail file: {detail}")

    G = build_dag(wf)

    name_map = _build_name_map(wf)
    use_name = (label_by.lower() == "name")
    H = _to_named_graph(G, name_map, use_name)
    log.info(f"DAG built: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")
    log.debug(f"Nodes: {list(H.nodes)}")
    log.debug(f"Edges: {list(H.edges)}")

    BOX_W, BOX_H = 0.28, 0.14
    pos = _layered_layout(H, horizontal=True, box_w=BOX_W, box_h=BOX_H)
    log.debug(f"Positions: {pos}")

    # Executed path (as names if requested)
    executed = _executed_path_from_detail(det)
    if executed and use_name:
        # if detail carried ids, map to names; if they are names already, no harm
        executed = [name_map.get(str(x), str(x)) for x in executed]

    _raw = _edge_list_from_path(executed) if len(executed) >= 2 else []
    hi_edges = [e for e in _raw if H.has_edge(*e)]

    # Prepare figure
    fig = plt.figure(figsize=(8.5, 6.2), dpi=180)
    ax = plt.gca()
    ax.set_axis_off()
    xs = [x for x, y in pos.values()]
    ys = [y for x, y in pos.values()]
    if xs and ys: 
        pad_x = BOX_W * 0.9
        pad_y = BOX_H * 0.9
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    ax.set_aspect("equal")

    MARGIN_PAD_PT = 2.0  # small extra padding in points
    margin_pt = data_to_points(BOX_W/2, BOX_H/2, ax) + MARGIN_PAD_PT

    src_margin_pt, tgt_margin_pt = _edge_margins(ax, BOX_W, BOX_H)

    # --- Draw base edges ---
    base_arts = nx.draw_networkx_edges(
        H, pos, ax=ax,
        width=1.4,
        alpha=0.7,                      
        arrows=True,
        arrowstyle='simple',
        arrowsize=28,
        edge_color=EDGE_BASE,          
        connectionstyle="arc3,rad=0.0",
        min_source_margin=src_margin_pt,
        min_target_margin=tgt_margin_pt,
    )

    if base_arts:
        for a in base_arts:
            try:
                a.set_zorder(2.2)
                a.set_clip_on(False)
            except Exception:
                pass

    # --- Draw highlighted edges (executed path) ---
    if hi_edges:
        hi_arts = nx.draw_networkx_edges(
            H, pos, ax=ax,
            edgelist=hi_edges,
            width=2.5, alpha=0.95,
            arrows=True,
            arrowstyle='simple',
            arrowsize=34,
            edge_color=EDGE_HI, 
            connectionstyle="arc3,rad=0.0",
            min_source_margin=src_margin_pt,
            min_target_margin=tgt_margin_pt,
        )

        if hi_arts:
            for a in hi_arts:
                try:
                    a.set_zorder(2.6) 
                    a.set_clip_on(False)
                except Exception:
                    pass

    # Draw nodes as rounded rectangles with colors
    # Assign colors: start -> PATH_START, end -> PATH_END, middle (if any) -> PATH_MID, others -> BASE_NODE
    path_set = set(executed)
    for n, (x, y) in pos.items():
        if not executed:
            fc = BASE_NODE
        else:
            if n == executed[0]:
                fc = PATH_START
            elif n == executed[-1]:
                fc = PATH_END
            elif n in path_set:
                fc = PATH_MID
            else:
                fc = BASE_NODE
        _draw_rounded_node(ax, (x, y), str(n), facecolor=fc)

    # Title & executed path label
    ttl = title or f"Workflow DAG: {Path(input).name}"
    ax.set_title(ttl, fontsize=13, pad=26)
    if executed:
        fig.text(0.02, 0.02, "Executed path: " + " -> ".join(executed), fontsize=9, ha="left", va="bottom")

    # Legend (optional)
    leg = None
    if show_legend and executed:
        legend_elems = [
            Line2D([0], [0], marker='s', color='w', label='Start (executed)', markerfacecolor=PATH_START, markersize=12),
            Line2D([0], [0], marker='s', color='w', label='Middle (executed)', markerfacecolor=PATH_MID, markersize=12),
            Line2D([0], [0], marker='s', color='w', label='End (executed)', markerfacecolor=PATH_END, markersize=12),
            Line2D([0], [0], marker='s', color='w', label='Other nodes', markerfacecolor=BASE_NODE, markersize=12),
            Line2D([0], [0], color=EDGE_HI, lw=3, label='Executed edges'),
        ]
        leg = fig.legend(
            handles=legend_elems,
            loc="upper left",
            bbox_to_anchor=(0.05, 0.90),
            frameon=False,
            fontsize=10,
            borderaxespad=0.0,
            ncol=min(len(legend_elems), 3),      
            handlelength=1.6, 
            columnspacing=1.8   
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    ax.margins(0.12) 
    plt.tight_layout()

    extra = dict(bbox_extra_artists=(leg,)) if 'leg' in locals() and leg else {}
    save_fig(plt.gcf(), out, bbox_inches="tight", pad_inches=0.05, **extra)
    log.info(f"[ok] wrote DAG to {out}")

if __name__ == "__main__":
    app()