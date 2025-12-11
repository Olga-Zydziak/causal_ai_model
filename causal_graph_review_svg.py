"""
causal_graph_review_svg.py

Interactive Human-in-the-Loop Causal Graph Review for Jupyter Lab.

Uses PURE SVG - no JavaScript, no iframe, works everywhere!

Requirements:
    pip install pandas numpy ipywidgets

Usage:
    from causal_graph_review_svg import review_causal_graph
    reviewer = review_causal_graph()

Version: 3.0.0 (Pure SVG - guaranteed to work!)
"""

from __future__ import annotations

import json
import html as html_module
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import pandas as pd
import numpy as np


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class EdgeStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class CausalEdge:
    """Represents a causal edge for review."""
    id: str
    source: str
    target: str
    discovered_strength: float
    ground_truth: float | None = None
    status: EdgeStatus = EdgeStatus.PENDING
    approved_strength: float | None = None
    is_confounded: bool = False
    confounders: list[str] = field(default_factory=list)
    is_spurious: bool = False
    rationale: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "discovered_strength": round(self.discovered_strength, 4),
            "ground_truth": round(self.ground_truth, 4) if self.ground_truth else None,
            "status": self.status.value,
            "approved_strength": round(self.approved_strength, 4) if self.approved_strength else None,
            "is_confounded": self.is_confounded,
            "confounders": self.confounders,
        }


# =============================================================================
# SVG GRAPH RENDERER
# =============================================================================

class SVGGraphRenderer:
    """Renders causal graph as pure SVG."""
    
    # Colors
    COLORS = {
        "bg": "#1e293b",
        "node_exo": "#64748b",
        "node_endo": "#3b82f6",
        "node_outcome": "#fbbf24",
        "edge_pending": "#94a3b8",
        "edge_approved": "#22c55e",
        "edge_rejected": "#ef4444",
        "edge_confounded": "#eab308",
        "text": "#ffffff",
        "text_dim": "#94a3b8",
    }
    
    def __init__(self, width: int = 700, height: int = 380):
        self.width = width
        self.height = height
    
    def render(
        self,
        nodes: dict[str, dict],
        edges: dict[str, "CausalEdge"],
        selected_edge_id: str | None = None,
    ) -> str:
        """Render graph as SVG string."""
        
        # Calculate node positions
        node_positions = self._layout_nodes(nodes, edges)
        
        # Build SVG
        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" '
            f'style="background:{self.COLORS["bg"]};border-radius:8px;">'
        ]
        
        # Defs for arrow markers
        svg_parts.append(self._render_defs())
        
        # Render edges first (behind nodes)
        for edge in edges.values():
            svg_parts.append(self._render_edge(edge, node_positions, selected_edge_id))
        
        # Render nodes
        for node_id, node_info in nodes.items():
            pos = node_positions[node_id]
            svg_parts.append(self._render_node(node_id, node_info, pos))
        
        # Legend
        svg_parts.append(self._render_legend())
        
        svg_parts.append('</svg>')
        
        return '\n'.join(svg_parts)
    
    def _layout_nodes(
        self,
        nodes: dict[str, dict],
        edges: dict[str, "CausalEdge"],
    ) -> dict[str, tuple[float, float]]:
        """Calculate node positions using hierarchical layout."""
        
        # Determine node types
        targets = {e.target for e in edges.values()}
        sources = {e.source for e in edges.values()}
        
        exo_nodes = []
        endo_nodes = []
        outcome_nodes = []
        
        for node_id in nodes:
            is_target = node_id in targets
            is_source = node_id in sources
            
            if node_id == "is_fraud" or (is_target and not is_source):
                outcome_nodes.append(node_id)
            elif is_target and is_source:
                endo_nodes.append(node_id)
            else:
                exo_nodes.append(node_id)
        
        positions = {}
        
        # Exogenous nodes on left
        x_exo = 100
        y_start = 50
        y_spacing = (self.height - 100) / max(len(exo_nodes), 1)
        for i, node_id in enumerate(sorted(exo_nodes)):
            positions[node_id] = (x_exo, y_start + i * y_spacing)
        
        # Endogenous nodes in middle
        x_endo = self.width // 2
        y_spacing = (self.height - 100) / max(len(endo_nodes), 1)
        for i, node_id in enumerate(sorted(endo_nodes)):
            positions[node_id] = (x_endo, y_start + 30 + i * y_spacing)
        
        # Outcome nodes on right
        x_outcome = self.width - 120
        y_spacing = (self.height - 100) / max(len(outcome_nodes), 1)
        for i, node_id in enumerate(sorted(outcome_nodes)):
            positions[node_id] = (x_outcome, self.height // 2)
        
        return positions
    
    def _render_defs(self) -> str:
        """Render SVG defs (arrow markers)."""
        return '''
        <defs>
            <marker id="arrow-pending" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#94a3b8"/>
            </marker>
            <marker id="arrow-approved" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#22c55e"/>
            </marker>
            <marker id="arrow-rejected" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#ef4444"/>
            </marker>
            <marker id="arrow-confounded" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#eab308"/>
            </marker>
        </defs>
        '''
    
    def _render_node(
        self,
        node_id: str,
        node_info: dict,
        pos: tuple[float, float],
    ) -> str:
        """Render a single node."""
        x, y = pos
        node_type = node_info.get("type", "exo")
        label = node_info.get("label", node_id)[:18]
        
        # Colors based on type
        if node_type == "outcome":
            fill = self.COLORS["node_outcome"]
            rx, ry = 8, 8
            w, h = 90, 45
        elif node_type == "endo":
            fill = self.COLORS["node_endo"]
            rx, ry = 6, 6
            w, h = 100, 36
        else:
            fill = self.COLORS["node_exo"]
            rx, ry = 6, 6
            w, h = 100, 32
        
        return f'''
        <g>
            <rect x="{x - w/2}" y="{y - h/2}" width="{w}" height="{h}" 
                  rx="{rx}" ry="{ry}" fill="{fill}" stroke="#475569" stroke-width="2"/>
            <text x="{x}" y="{y + 4}" text-anchor="middle" 
                  fill="{self.COLORS["text"]}" font-size="11" font-family="Arial, sans-serif" font-weight="bold">
                {html_module.escape(label)}
            </text>
        </g>
        '''
    
    def _render_edge(
        self,
        edge: "CausalEdge",
        positions: dict[str, tuple[float, float]],
        selected_edge_id: str | None,
    ) -> str:
        """Render a single edge."""
        if edge.source not in positions or edge.target not in positions:
            return ""
        
        x1, y1 = positions[edge.source]
        x2, y2 = positions[edge.target]
        
        # Offset from node center
        dx = x2 - x1
        dy = y2 - y1
        dist = math.sqrt(dx*dx + dy*dy)
        if dist == 0:
            return ""
        
        # Normalize
        nx, ny = dx/dist, dy/dist
        
        # Start/end offsets (from node edges)
        x1 += nx * 55
        y1 += ny * 20
        x2 -= nx * 55
        y2 -= ny * 20
        
        # Determine color and style
        if edge.status == EdgeStatus.APPROVED:
            color = self.COLORS["edge_approved"]
            marker = "arrow-approved"
            width = 3
            dash = ""
        elif edge.status == EdgeStatus.REJECTED:
            color = self.COLORS["edge_rejected"]
            marker = "arrow-rejected"
            width = 2
            dash = "stroke-dasharray='6,4'"
        elif edge.is_confounded:
            color = self.COLORS["edge_confounded"]
            marker = "arrow-confounded"
            width = 2.5
            dash = ""
        else:
            color = self.COLORS["edge_pending"]
            marker = "arrow-pending"
            width = 2
            dash = ""
        
        # Highlight selected
        if selected_edge_id == edge.id:
            width += 2
        
        # Slight curve for overlapping edges
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        curve_offset = 15 if abs(dy) < 50 else 0
        ctrl_x = mid_x - ny * curve_offset
        ctrl_y = mid_y + nx * curve_offset
        
        # Strength label for approved edges
        label_svg = ""
        if edge.status == EdgeStatus.APPROVED and edge.approved_strength is not None:
            label_svg = f'''
            <text x="{mid_x}" y="{mid_y - 8}" text-anchor="middle" 
                  fill="{self.COLORS["text_dim"]}" font-size="9" font-family="Arial, sans-serif">
                β={edge.approved_strength:+.2f}
            </text>
            '''
        
        # Status indicator
        status_icon = ""
        if edge.status == EdgeStatus.APPROVED:
            status_icon = f'<circle cx="{mid_x}" cy="{mid_y}" r="8" fill="#22c55e"/><text x="{mid_x}" y="{mid_y+4}" text-anchor="middle" fill="white" font-size="10">✓</text>'
        elif edge.status == EdgeStatus.REJECTED:
            status_icon = f'<circle cx="{mid_x}" cy="{mid_y}" r="8" fill="#ef4444"/><text x="{mid_x}" y="{mid_y+4}" text-anchor="middle" fill="white" font-size="10">✗</text>'
        elif edge.is_confounded:
            status_icon = f'<circle cx="{mid_x + 15}" cy="{mid_y}" r="7" fill="#eab308"/><text x="{mid_x + 15}" y="{mid_y+4}" text-anchor="middle" fill="#1e293b" font-size="9" font-weight="bold">!</text>'
        
        return f'''
        <g>
            <path d="M {x1} {y1} Q {ctrl_x} {ctrl_y} {x2} {y2}" 
                  stroke="{color}" stroke-width="{width}" fill="none" 
                  marker-end="url(#{marker})" {dash}/>
            {status_icon}
            {label_svg}
        </g>
        '''
    
    def _render_legend(self) -> str:
        """Render legend."""
        y = self.height - 25
        return f'''
        <g font-size="10" font-family="Arial, sans-serif">
            <rect x="10" y="{y-12}" width="12" height="12" fill="{self.COLORS["node_exo"]}" rx="2"/>
            <text x="26" y="{y}" fill="{self.COLORS["text_dim"]}">Exogenous</text>
            
            <rect x="100" y="{y-12}" width="12" height="12" fill="{self.COLORS["node_endo"]}" rx="2"/>
            <text x="116" y="{y}" fill="{self.COLORS["text_dim"]}">Endogenous</text>
            
            <rect x="200" y="{y-12}" width="12" height="12" fill="{self.COLORS["node_outcome"]}" rx="2"/>
            <text x="216" y="{y}" fill="{self.COLORS["text_dim"]}">Outcome</text>
            
            <line x1="290" y1="{y-6}" x2="310" y2="{y-6}" stroke="{self.COLORS["edge_approved"]}" stroke-width="3"/>
            <text x="315" y="{y}" fill="{self.COLORS["text_dim"]}">Approved</text>
            
            <line x1="380" y1="{y-6}" x2="400" y2="{y-6}" stroke="{self.COLORS["edge_rejected"]}" stroke-width="2" stroke-dasharray="4,3"/>
            <text x="405" y="{y}" fill="{self.COLORS["text_dim"]}">Rejected</text>
            
            <line x1="465" y1="{y-6}" x2="485" y2="{y-6}" stroke="{self.COLORS["edge_confounded"]}" stroke-width="2.5"/>
            <text x="490" y="{y}" fill="{self.COLORS["text_dim"]}">Confounded</text>
        </g>
        '''


# =============================================================================
# MAIN REVIEWER CLASS
# =============================================================================

class CausalGraphReviewer:
    """Interactive Causal Graph Review Interface using pure SVG."""
    
    def __init__(
        self,
        edges: list[CausalEdge],
        reviewer_name: str = "",
        data: pd.DataFrame | None = None,
    ):
        self.edges = {e.id: e for e in edges}
        self.reviewer_name = reviewer_name
        self.data = data
        self._selected_edge_id: str | None = None
        
        # Collect nodes
        self.nodes: dict[str, dict] = {}
        targets = {e.target for e in edges}
        sources = {e.source for e in edges}
        
        all_node_ids = sources | targets
        for node_id in all_node_ids:
            is_target = node_id in targets
            is_source = node_id in sources
            
            if node_id == "is_fraud" or (is_target and not is_source):
                node_type = "outcome"
            elif is_target and is_source:
                node_type = "endo"
            else:
                node_type = "exo"
            
            label = node_id.replace("_", " ").title()
            if len(label) > 16:
                label = label[:14] + ".."
            
            self.nodes[node_id] = {"type": node_type, "label": label}
        
        # Renderer
        self._renderer = SVGGraphRenderer()
        
        # Build UI
        self._build_ui()
    
    @classmethod
    def from_discovery_result(
        cls,
        discovery_path: str | Path,
        ground_truth_path: str | Path | None = None,
        data_path: str | Path | None = None,
        known_confounders: dict[str, list[str]] | None = None,
    ) -> "CausalGraphReviewer":
        """Create reviewer from discovery result JSON."""
        with open(discovery_path) as f:
            discovery = json.load(f)
        
        gt_map: dict[str, float] = {}
        if ground_truth_path and Path(ground_truth_path).exists():
            with open(ground_truth_path) as f:
                gt_data = json.load(f)
            for edge in gt_data.get("ground_truth", {}).get("causal_edges", []):
                key = f"{edge['source']}_to_{edge['target']}"
                gt_map[key] = edge["true_strength"]
        
        data = None
        if data_path and Path(data_path).exists():
            data = pd.read_csv(data_path)
        
        edges_data = discovery.get("discovered_graph", {}).get("edges", [])
        edges = []
        
        for i, e in enumerate(edges_data):
            source = e["source"]
            target = e["target"]
            edge_key = f"{source}_to_{target}"
            
            gt_strength = gt_map.get(edge_key)
            is_spurious = gt_strength is None and len(gt_map) > 0
            
            confounders = []
            is_confounded = False
            if known_confounders and edge_key in known_confounders:
                confounders = known_confounders[edge_key]
                is_confounded = True
            
            edge = CausalEdge(
                id=f"e{i+1}",
                source=source,
                target=target,
                discovered_strength=e.get("strength", 1.0),
                ground_truth=gt_strength,
                is_confounded=is_confounded,
                confounders=confounders,
                is_spurious=is_spurious,
            )
            edges.append(edge)
        
        return cls(edges=edges, data=data)
    
    def _build_ui(self) -> None:
        """Build UI components."""
        # Header
        self._header = widgets.HTML(value=self._render_header())
        
        # Stats
        self._stats_bar = widgets.HTML(value=self._render_stats())
        
        # Graph
        self._graph_html = widgets.HTML(value=self._render_graph())
        
        # Edge selector
        edge_options = [("-- Select edge to review --", None)] + [
            (f"{'○' if e.status==EdgeStatus.PENDING else '✓' if e.status==EdgeStatus.APPROVED else '✗'} {e.source} → {e.target}", e.id) 
            for e in self.edges.values()
        ]
        self._edge_selector = widgets.Dropdown(
            options=edge_options,
            value=None,
            layout=widgets.Layout(width="100%"),
        )
        self._edge_selector.observe(self._on_edge_select, names="value")
        
        # Details
        self._details_html = widgets.HTML(value=self._render_details())
        
        # Inputs
        self._reviewer_input = widgets.Text(
            value=self.reviewer_name,
            placeholder="Your name...",
            layout=widgets.Layout(width="100%"),
        )
        self._reviewer_input.observe(lambda c: setattr(self, 'reviewer_name', c["new"]), names="value")
        
        self._strength_input = widgets.BoundedFloatText(
            value=0.0, min=-2.0, max=2.0, step=0.01,
            layout=widgets.Layout(width="100%"),
        )
        
        self._confounded_cb = widgets.Checkbox(
            value=False,
            description="Mark as confounded",
            indent=False,
        )
        
        # Buttons
        self._approve_btn = widgets.Button(description="✓ Approve", button_style="success", layout=widgets.Layout(width="48%"))
        self._reject_btn = widgets.Button(description="✗ Reject", button_style="danger", layout=widgets.Layout(width="48%"))
        self._auto_btn = widgets.Button(description="🚀 Auto-approve (Ground Truth)", button_style="primary", layout=widgets.Layout(width="100%"))
        self._export_btn = widgets.Button(description="📤 Export JSON", button_style="info", layout=widgets.Layout(width="100%"))
        
        self._approve_btn.on_click(self._on_approve)
        self._reject_btn.on_click(self._on_reject)
        self._auto_btn.on_click(self._on_auto_approve)
        self._export_btn.on_click(self._on_export)
        
        self._export_output = widgets.Output()
        
        # Layout
        btn_row = widgets.HBox([self._approve_btn, self._reject_btn])
        
        control_panel = widgets.VBox([
            widgets.HTML("<div style='font-weight:bold;color:#60a5fa;margin-bottom:8px;'>📋 Edge Review</div>"),
            widgets.HTML("<div style='color:#94a3b8;font-size:11px;'>Select edge:</div>"),
            self._edge_selector,
            self._details_html,
            widgets.HTML("<hr style='border-color:#334155;'>"),
            widgets.HTML("<div style='color:#94a3b8;font-size:11px;'>Reviewer:</div>"),
            self._reviewer_input,
            widgets.HTML("<div style='color:#94a3b8;font-size:11px;'>Strength (β):</div>"),
            self._strength_input,
            self._confounded_cb,
            btn_row,
            widgets.HTML("<hr style='border-color:#334155;'>"),
            self._auto_btn,
            self._export_btn,
        ], layout=widgets.Layout(width="280px", padding="10px", border="1px solid #475569", border_radius="8px"))
        
        main_row = widgets.HBox([
            widgets.VBox([self._graph_html], layout=widgets.Layout(flex="1")),
            control_panel,
        ], layout=widgets.Layout(gap="10px"))
        
        self._container = widgets.VBox([
            self._header,
            self._stats_bar,
            main_row,
            self._export_output,
        ])
    
    def _render_header(self) -> str:
        return """
        <div style="background:linear-gradient(135deg,#1e3a5f,#1e293b);padding:12px;border-radius:8px;margin-bottom:10px;">
            <h2 style="margin:0;color:#60a5fa;font-size:18px;">🔗 Causal Graph Review Interface</h2>
            <p style="margin:4px 0 0 0;color:#94a3b8;font-size:12px;">Select edges from dropdown • Approve or reject each edge</p>
        </div>
        """
    
    def _render_stats(self) -> str:
        approved = sum(1 for e in self.edges.values() if e.status == EdgeStatus.APPROVED)
        rejected = sum(1 for e in self.edges.values() if e.status == EdgeStatus.REJECTED)
        pending = sum(1 for e in self.edges.values() if e.status == EdgeStatus.PENDING)
        confounded = sum(1 for e in self.edges.values() if e.is_confounded and e.status != EdgeStatus.REJECTED)
        total = len(self.edges)
        progress = int((approved + rejected) / total * 100) if total else 0
        
        return f"""
        <div style="background:#334155;padding:10px;border-radius:8px;margin-bottom:10px;display:flex;gap:20px;font-size:13px;">
            <span style="color:#94a3b8;">○ Pending: <b style="color:white;">{pending}</b></span>
            <span style="color:#94a3b8;">● Approved: <b style="color:#22c55e;">{approved}</b></span>
            <span style="color:#94a3b8;">● Rejected: <b style="color:#ef4444;">{rejected}</b></span>
            <span style="color:#94a3b8;">● Confounded: <b style="color:#eab308;">{confounded}</b></span>
            <span style="margin-left:auto;color:#94a3b8;">Progress: <b style="color:white;">{progress}%</b></span>
        </div>
        """
    
    def _render_graph(self) -> str:
        return self._renderer.render(self.nodes, self.edges, self._selected_edge_id)
    
    def _render_details(self) -> str:
        if not self._selected_edge_id:
            return '<div style="text-align:center;padding:15px;color:#64748b;">👆 Select an edge</div>'
        
        edge = self.edges.get(self._selected_edge_id)
        if not edge:
            return ""
        
        status_colors = {
            EdgeStatus.PENDING: ("#6b7280", "PENDING"),
            EdgeStatus.APPROVED: ("#22c55e", "APPROVED"),
            EdgeStatus.REJECTED: ("#ef4444", "REJECTED"),
        }
        color, status_text = status_colors.get(edge.status, ("#6b7280", "?"))
        
        warnings = ""
        if edge.is_spurious:
            warnings += '<div style="background:#7f1d1d;color:#fca5a5;padding:4px 6px;border-radius:4px;margin:4px 0;font-size:10px;">⚠️ SPURIOUS</div>'
        if edge.is_confounded:
            warnings += '<div style="background:#713f12;color:#fde047;padding:4px 6px;border-radius:4px;margin:4px 0;font-size:10px;">⚠️ CONFOUNDED</div>'
        
        gt = f'<div style="display:flex;justify-content:space-between;font-size:11px;"><span style="color:#94a3b8;">Ground Truth:</span><span style="color:#60a5fa;">{edge.ground_truth:+.3f}</span></div>' if edge.ground_truth else ""
        
        return f"""
        <div style="background:#1e293b;padding:8px;border-radius:6px;margin:8px 0;font-size:11px;">
            <div style="text-align:center;margin-bottom:6px;">
                <span style="color:#fbbf24;">{edge.source}</span> → <span style="color:#4ade80;">{edge.target}</span>
            </div>
            {warnings}
            <div style="display:flex;justify-content:space-between;"><span style="color:#94a3b8;">Discovered:</span><span>{edge.discovered_strength:+.3f}</span></div>
            {gt}
            <div style="display:flex;justify-content:space-between;"><span style="color:#94a3b8;">Status:</span><span style="color:{color};font-weight:bold;">{status_text}</span></div>
        </div>
        """
    
    def _on_edge_select(self, change):
        self._selected_edge_id = change["new"]
        self._details_html.value = self._render_details()
        self._graph_html.value = self._render_graph()
        
        if self._selected_edge_id:
            edge = self.edges[self._selected_edge_id]
            self._strength_input.value = edge.approved_strength or edge.ground_truth or edge.discovered_strength
            self._confounded_cb.value = edge.is_confounded
    
    def _on_approve(self, btn):
        if not self._selected_edge_id:
            with self._export_output:
                clear_output()
                display(HTML('<p style="color:#fbbf24;">⚠️ Select an edge first!</p>'))
            return
        edge = self.edges[self._selected_edge_id]
        edge.status = EdgeStatus.APPROVED
        edge.approved_strength = self._strength_input.value
        edge.is_confounded = self._confounded_cb.value
        
        # Feedback
        with self._export_output:
            clear_output()
            display(HTML(f'<p style="color:#22c55e;">✓ Approved: {edge.source} → {edge.target} (β={edge.approved_strength:.4f})</p>'))
        
        self._refresh()
    
    def _on_reject(self, btn):
        if not self._selected_edge_id:
            with self._export_output:
                clear_output()
                display(HTML('<p style="color:#fbbf24;">⚠️ Select an edge first!</p>'))
            return
        edge = self.edges[self._selected_edge_id]
        edge.status = EdgeStatus.REJECTED
        
        # Feedback
        with self._export_output:
            clear_output()
            display(HTML(f'<p style="color:#ef4444;">✗ Rejected: {edge.source} → {edge.target}</p>'))
        
        self._refresh()
    
    def _on_auto_approve(self, btn):
        n_approved = 0
        n_rejected = 0
        for edge in self.edges.values():
            if edge.status == EdgeStatus.PENDING:
                if edge.is_spurious:
                    edge.status = EdgeStatus.REJECTED
                    n_rejected += 1
                else:
                    edge.status = EdgeStatus.APPROVED
                    edge.approved_strength = edge.ground_truth or edge.discovered_strength
                    n_approved += 1
        
        # Feedback
        with self._export_output:
            clear_output()
            display(HTML(f'<p style="color:#22c55e;">✓ Auto-approved {n_approved} edges, rejected {n_rejected} spurious</p>'))
        
        self._refresh()
    
    def _on_export(self, btn):
        data = self.export_approved()
        json_str = json.dumps(data, indent=2)
        with self._export_output:
            clear_output()
            display(HTML(f'''
            <div style="background:#1e293b;padding:12px;border-radius:8px;margin-top:10px;border:1px solid #475569;">
                <h4 style="margin:0 0 8px 0;color:#60a5fa;">📤 Exported JSON</h4>
                <textarea readonly style="width:100%;height:150px;background:#0f172a;color:#4ade80;border:1px solid #334155;border-radius:4px;padding:8px;font-family:monospace;font-size:10px;">{html_module.escape(json_str)}</textarea>
            </div>
            '''))
    
    def _refresh(self):
        self._stats_bar.value = self._render_stats()
        self._details_html.value = self._render_details()
        self._graph_html.value = self._render_graph()
        
        edge_options = [("-- Select edge --", None)] + [
            (f"{'○' if e.status==EdgeStatus.PENDING else '✓' if e.status==EdgeStatus.APPROVED else '✗'} {e.source} → {e.target}", e.id) 
            for e in self.edges.values()
        ]
        self._edge_selector.options = edge_options
        self._edge_selector.value = self._selected_edge_id
    
    def display(self):
        display(self._container)
    
    def export_approved(self) -> dict:
        return {
            "metadata": {
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "reviewed_by": self.reviewer_name or "Anonymous",
                "n_approved": sum(1 for e in self.edges.values() if e.status == EdgeStatus.APPROVED),
                "n_rejected": sum(1 for e in self.edges.values() if e.status == EdgeStatus.REJECTED),
            },
            "approved_edges": [e.to_dict() for e in self.edges.values() if e.status == EdgeStatus.APPROVED],
            "rejected_edges": [{"source": e.source, "target": e.target} for e in self.edges.values() if e.status == EdgeStatus.REJECTED],
        }
    
    def save_approved(self, path: str | Path):
        """Save approved graph to JSON file."""
        data = self.export_approved()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        n_approved = data["metadata"]["n_approved"]
        n_rejected = data["metadata"]["n_rejected"]
        
        if n_approved == 0 and n_rejected == 0:
            print(f"⚠️ Warning: No edges approved or rejected!")
            print(f"   Did you click 'Approve' or 'Auto-approve' first?")
            print(f"   Total edges in graph: {len(self.edges)}")
        else:
            print(f"✓ Saved: {path}")
            print(f"  Approved: {n_approved}, Rejected: {n_rejected}")


# =============================================================================
# QUICK START
# =============================================================================

def review_causal_graph(
    discovery_path: str = "discovery_result.json",
    ground_truth_path: str | None = "ground_truth.json",
    data_path: str | None = None,
) -> CausalGraphReviewer:
    """Launch the review interface."""
    known_confounders = {"transaction_velocity_24h_to_is_fraud": ["criminal_intent"]}
    
    reviewer = CausalGraphReviewer.from_discovery_result(
        discovery_path=discovery_path,
        ground_truth_path=ground_truth_path if ground_truth_path and Path(ground_truth_path).exists() else None,
        data_path=data_path,
        known_confounders=known_confounders,
    )
    reviewer.display()
    return reviewer


if __name__ == "__main__":
    print("from causal_graph_review_svg import review_causal_graph")
    print("reviewer = review_causal_graph()")
