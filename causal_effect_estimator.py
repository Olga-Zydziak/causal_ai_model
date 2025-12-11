"""
causal_effect_estimator.py

Causal Effect Estimation Engine - oblicza ATE (Average Treatment Effect)
i CATE (Conditional Average Treatment Effect) na podstawie zatwierdzonego
grafu przyczynowego.

Integracja:
    from causal_effect_estimator import CausalEffectEstimator
    
    estimator = CausalEffectEstimator.from_files(
        approved_graph_path="approved_graph.json",
        data_path="fraud_data.csv",
    )
    report = estimator.estimate_all()
    report.display()

Wymagania:
    pip install pandas numpy scipy statsmodels

Author: Causal AI Engine
Version: 1.0.0
PEP: 8, 257, 484, 585, 604
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd
from scipy import stats

# Suppress warnings for clean output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class EffectSignificance(Enum):
    """Statistical significance level."""
    HIGHLY_SIGNIFICANT = "***"  # p < 0.001
    SIGNIFICANT = "**"          # p < 0.01
    MARGINALLY_SIGNIFICANT = "*"  # p < 0.05
    NOT_SIGNIFICANT = ""        # p >= 0.05


@dataclass(frozen=True, slots=True)
class CausalEdgeInfo:
    """Information about a causal edge from approved graph."""
    source: str
    target: str
    approved_strength: float
    is_confounded: bool
    confounders: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ATEResult:
    """
    Average Treatment Effect estimation result.
    
    Attributes:
        treatment: Name of treatment variable (cause).
        outcome: Name of outcome variable (effect).
        ate: Estimated Average Treatment Effect.
        std_error: Standard error of the estimate.
        ci_lower: Lower bound of 95% confidence interval.
        ci_upper: Upper bound of 95% confidence interval.
        p_value: P-value for H0: ATE = 0.
        t_statistic: T-statistic for the test.
        sample_size: Number of observations used.
        r_squared: R² of the regression model.
        method: Estimation method used.
    """
    treatment: str
    outcome: str
    ate: float
    std_error: float
    ci_lower: float
    ci_upper: float
    p_value: float
    t_statistic: float
    sample_size: int
    r_squared: float
    method: str = "OLS Regression"
    
    @property
    def significance(self) -> EffectSignificance:
        """Determine statistical significance level."""
        if self.p_value < 0.001:
            return EffectSignificance.HIGHLY_SIGNIFICANT
        if self.p_value < 0.01:
            return EffectSignificance.SIGNIFICANT
        if self.p_value < 0.05:
            return EffectSignificance.MARGINALLY_SIGNIFICANT
        return EffectSignificance.NOT_SIGNIFICANT
    
    @property
    def is_significant(self) -> bool:
        """Check if effect is statistically significant at α=0.05."""
        return self.p_value < 0.05
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "treatment": self.treatment,
            "outcome": self.outcome,
            "ate": round(self.ate, 6),
            "std_error": round(self.std_error, 6),
            "confidence_interval": {
                "lower": round(self.ci_lower, 6),
                "upper": round(self.ci_upper, 6),
                "level": 0.95,
            },
            "p_value": round(self.p_value, 6),
            "t_statistic": round(self.t_statistic, 4),
            "is_significant": self.is_significant,
            "significance_level": self.significance.value,
            "sample_size": self.sample_size,
            "r_squared": round(self.r_squared, 4),
            "method": self.method,
        }


@dataclass(frozen=True, slots=True)
class CATEResult:
    """
    Conditional Average Treatment Effect estimation result.
    
    Attributes:
        treatment: Name of treatment variable.
        outcome: Name of outcome variable.
        subgroup_name: Name of the subgroup.
        subgroup_description: Human-readable description.
        cate: Estimated Conditional ATE for this subgroup.
        std_error: Standard error of the estimate.
        ci_lower: Lower bound of 95% CI.
        ci_upper: Upper bound of 95% CI.
        p_value: P-value for H0: CATE = 0.
        sample_size: Number of observations in subgroup.
        subgroup_proportion: Proportion of total sample.
    """
    treatment: str
    outcome: str
    subgroup_name: str
    subgroup_description: str
    cate: float
    std_error: float
    ci_lower: float
    ci_upper: float
    p_value: float
    sample_size: int
    subgroup_proportion: float
    
    @property
    def is_significant(self) -> bool:
        """Check if effect is statistically significant at α=0.05."""
        return self.p_value < 0.05
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "treatment": self.treatment,
            "outcome": self.outcome,
            "subgroup_name": self.subgroup_name,
            "subgroup_description": self.subgroup_description,
            "cate": round(self.cate, 6),
            "std_error": round(self.std_error, 6),
            "confidence_interval": {
                "lower": round(self.ci_lower, 6),
                "upper": round(self.ci_upper, 6),
            },
            "p_value": round(self.p_value, 6),
            "is_significant": self.is_significant,
            "sample_size": self.sample_size,
            "subgroup_proportion": round(self.subgroup_proportion, 4),
        }


@dataclass
class SubgroupDefinition:
    """Definition of a subgroup for CATE estimation."""
    name: str
    description: str
    condition: Callable[[pd.DataFrame], pd.Series]


# =============================================================================
# EFFECT REPORT
# =============================================================================

@dataclass
class CausalEffectReport:
    """
    Complete report of causal effect estimation.
    
    Contains all ATE and CATE results with metadata.
    """
    ate_results: list[ATEResult]
    cate_results: list[CATEResult]
    outcome_variable: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_sample_size: int = 0
    
    def get_ate_ranking(self) -> list[ATEResult]:
        """Get ATE results ranked by absolute effect size."""
        return sorted(self.ate_results, key=lambda x: abs(x.ate), reverse=True)
    
    def get_significant_effects(self) -> list[ATEResult]:
        """Get only statistically significant ATE results."""
        return [r for r in self.ate_results if r.is_significant]
    
    def display(self) -> None:
        """Display formatted report in Jupyter/console."""
        from IPython.display import display, HTML
        
        html = self._generate_html_report()
        display(HTML(html))
    
    def _generate_html_report(self) -> str:
        """Generate HTML report."""
        # Header
        html_parts = [
            '<div style="font-family:Arial,sans-serif;max-width:1000px;">',
            '<div style="background:linear-gradient(135deg,#1e3a5f,#0f172a);padding:20px;border-radius:8px;margin-bottom:20px;">',
            '<h2 style="margin:0;color:#60a5fa;">📊 Causal Effect Estimation Report</h2>',
            f'<p style="margin:8px 0 0 0;color:#94a3b8;">Outcome: <b style="color:#fbbf24;">{self.outcome_variable}</b> | '
            f'Sample size: <b style="color:white;">{self.total_sample_size:,}</b> | '
            f'Generated: {self.timestamp[:19]}</p>',
            '</div>',
        ]
        
        # ATE Table
        html_parts.append('<h3 style="color:#e2e8f0;margin-bottom:10px;">🎯 Average Treatment Effects (ATE)</h3>')
        html_parts.append('<p style="color:#94a3b8;font-size:13px;margin-bottom:15px;">'
                         'Shows how much each factor influences the outcome on average.</p>')
        html_parts.append(self._generate_ate_table())
        
        # Top factors
        html_parts.append('<h3 style="color:#e2e8f0;margin:25px 0 10px 0;">🏆 Top Impact Factors</h3>')
        html_parts.append(self._generate_ranking_bars())
        
        # CATE section with visualizations
        if self.cate_results:
            html_parts.append('<h3 style="color:#e2e8f0;margin:25px 0 10px 0;">🔍 Conditional Effects (CATE by Subgroup)</h3>')
            html_parts.append('<p style="color:#94a3b8;font-size:13px;margin-bottom:15px;">'
                             'Shows how effects vary across different customer segments.</p>')
            
            # CATE Heatmap
            html_parts.append(self._generate_cate_heatmap())
            
            # CATE Bar Charts
            html_parts.append(self._generate_cate_bar_charts())
            
            # CATE Table (detailed)
            html_parts.append('<h4 style="color:#94a3b8;margin:25px 0 10px 0;">📋 Detailed CATE Values</h4>')
            html_parts.append(self._generate_cate_table())
        
        # Interpretation guide
        html_parts.append(self._generate_interpretation_guide())
        
        html_parts.append('</div>')
        
        return '\n'.join(html_parts)
    
    def _generate_ate_table(self) -> str:
        """Generate ATE results table."""
        rows = []
        for r in self.get_ate_ranking():
            sig_color = "#22c55e" if r.is_significant else "#6b7280"
            sig_stars = r.significance.value
            effect_color = "#ef4444" if r.ate > 0 else "#22c55e" if r.ate < 0 else "#94a3b8"
            
            rows.append(f'''
            <tr style="border-bottom:1px solid #334155;">
                <td style="padding:10px;color:#fbbf24;">{r.treatment}</td>
                <td style="padding:10px;color:{effect_color};font-weight:bold;font-family:monospace;">
                    {r.ate:+.4f}{sig_stars}
                </td>
                <td style="padding:10px;color:#94a3b8;font-family:monospace;font-size:12px;">
                    [{r.ci_lower:+.4f}, {r.ci_upper:+.4f}]
                </td>
                <td style="padding:10px;color:{sig_color};font-family:monospace;">
                    {r.p_value:.4f}
                </td>
                <td style="padding:10px;color:#64748b;text-align:right;">{r.sample_size:,}</td>
            </tr>
            ''')
        
        return f'''
        <table style="width:100%;border-collapse:collapse;background:#1e293b;border-radius:8px;overflow:hidden;">
            <thead>
                <tr style="background:#334155;">
                    <th style="padding:12px;text-align:left;color:#94a3b8;font-weight:normal;">Treatment</th>
                    <th style="padding:12px;text-align:left;color:#94a3b8;font-weight:normal;">ATE (β)</th>
                    <th style="padding:12px;text-align:left;color:#94a3b8;font-weight:normal;">95% CI</th>
                    <th style="padding:12px;text-align:left;color:#94a3b8;font-weight:normal;">p-value</th>
                    <th style="padding:12px;text-align:right;color:#94a3b8;font-weight:normal;">N</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        '''
    
    def _generate_ranking_bars(self) -> str:
        """Generate visual ranking bars."""
        ranked = self.get_ate_ranking()[:5]  # Top 5
        if not ranked:
            return ""
        
        max_abs = max(abs(r.ate) for r in ranked) or 1
        
        bars = []
        for i, r in enumerate(ranked, 1):
            width = int(abs(r.ate) / max_abs * 200)
            color = "#ef4444" if r.ate > 0 else "#22c55e"
            direction = "→ increases" if r.ate > 0 else "→ decreases"
            sig = "✓" if r.is_significant else ""
            
            bars.append(f'''
            <div style="display:flex;align-items:center;margin:8px 0;">
                <span style="width:25px;color:#64748b;font-size:12px;">#{i}</span>
                <span style="width:180px;color:#e2e8f0;font-size:13px;">{r.treatment[:25]}</span>
                <div style="width:220px;background:#334155;border-radius:4px;height:20px;margin:0 10px;">
                    <div style="width:{width}px;background:{color};height:100%;border-radius:4px;"></div>
                </div>
                <span style="color:{color};font-family:monospace;width:80px;">{r.ate:+.3f}</span>
                <span style="color:#64748b;font-size:11px;width:100px;">{direction}</span>
                <span style="color:#22c55e;">{sig}</span>
            </div>
            ''')
        
        return f'<div style="background:#1e293b;padding:15px;border-radius:8px;">{"".join(bars)}</div>'
    
    def _generate_cate_table(self) -> str:
        """Generate CATE results table."""
        if not self.cate_results:
            return ""
        
        # Group by treatment
        by_treatment: dict[str, list[CATEResult]] = {}
        for r in self.cate_results:
            key = r.treatment
            if key not in by_treatment:
                by_treatment[key] = []
            by_treatment[key].append(r)
        
        tables = []
        for treatment, results in by_treatment.items():
            rows = []
            for r in results:
                effect_color = "#ef4444" if r.cate > 0 else "#22c55e" if r.cate < 0 else "#94a3b8"
                sig = "✓" if r.is_significant else ""
                
                rows.append(f'''
                <tr style="border-bottom:1px solid #334155;">
                    <td style="padding:8px;color:#e2e8f0;">{r.subgroup_name}</td>
                    <td style="padding:8px;color:#64748b;font-size:12px;">{r.subgroup_description}</td>
                    <td style="padding:8px;color:{effect_color};font-weight:bold;font-family:monospace;">
                        {r.cate:+.4f}
                    </td>
                    <td style="padding:8px;color:#22c55e;">{sig}</td>
                    <td style="padding:8px;color:#64748b;text-align:right;">{r.sample_size:,} ({r.subgroup_proportion:.0%})</td>
                </tr>
                ''')
            
            tables.append(f'''
            <div style="margin-bottom:20px;">
                <h4 style="color:#fbbf24;margin:0 0 10px 0;">Effect of: {treatment}</h4>
                <table style="width:100%;border-collapse:collapse;background:#1e293b;border-radius:6px;overflow:hidden;font-size:13px;">
                    <thead>
                        <tr style="background:#334155;">
                            <th style="padding:10px;text-align:left;color:#94a3b8;font-weight:normal;">Subgroup</th>
                            <th style="padding:10px;text-align:left;color:#94a3b8;font-weight:normal;">Description</th>
                            <th style="padding:10px;text-align:left;color:#94a3b8;font-weight:normal;">CATE</th>
                            <th style="padding:10px;text-align:left;color:#94a3b8;font-weight:normal;">Sig.</th>
                            <th style="padding:10px;text-align:right;color:#94a3b8;font-weight:normal;">N (%)</th>
                        </tr>
                    </thead>
                    <tbody>{''.join(rows)}</tbody>
                </table>
            </div>
            ''')
        
        return ''.join(tables)
    
    def _generate_cate_heatmap(self) -> str:
        """Generate CATE heatmap visualization."""
        if not self.cate_results:
            return ""
        
        # Organize data: treatments × subgroups
        treatments = sorted(set(r.treatment for r in self.cate_results))
        subgroups = sorted(set(r.subgroup_name for r in self.cate_results))
        
        # Create lookup
        cate_lookup: dict[tuple[str, str], float] = {}
        for r in self.cate_results:
            cate_lookup[(r.treatment, r.subgroup_name)] = r.cate
        
        # Find min/max for color scaling
        all_cates = [r.cate for r in self.cate_results if not np.isnan(r.cate)]
        if not all_cates:
            return ""
        
        max_abs = max(abs(min(all_cates)), abs(max(all_cates))) or 1
        
        def get_color(value: float) -> str:
            """Get color for heatmap cell."""
            if np.isnan(value):
                return "#1e293b"
            
            # Normalize to -1 to 1
            normalized = value / max_abs
            normalized = max(-1, min(1, normalized))
            
            if normalized > 0:
                # Red gradient for positive (increases fraud)
                intensity = int(normalized * 180)
                return f"rgb({120 + intensity}, {60 - int(normalized * 40)}, {60 - int(normalized * 40)})"
            else:
                # Green gradient for negative (decreases fraud)
                intensity = int(-normalized * 180)
                return f"rgb({60 - int(-normalized * 40)}, {120 + intensity}, {60 - int(-normalized * 40)})"
        
        # Build heatmap
        header_cells = ['<th style="padding:10px;text-align:left;color:#94a3b8;font-weight:normal;min-width:120px;">Treatment</th>']
        for subgroup in subgroups:
            # Truncate long names
            display_name = subgroup[:15] + "..." if len(subgroup) > 15 else subgroup
            header_cells.append(
                f'<th style="padding:10px;text-align:center;color:#94a3b8;font-weight:normal;font-size:11px;min-width:80px;">'
                f'{display_name}</th>'
            )
        
        rows = []
        for treatment in treatments:
            # Truncate treatment name
            display_treatment = treatment[:25] + "..." if len(treatment) > 25 else treatment
            cells = [f'<td style="padding:10px;color:#fbbf24;font-size:12px;">{display_treatment}</td>']
            
            for subgroup in subgroups:
                value = cate_lookup.get((treatment, subgroup), np.nan)
                color = get_color(value)
                text_color = "#fff" if not np.isnan(value) else "#64748b"
                display_value = f"{value:+.3f}" if not np.isnan(value) else "N/A"
                
                cells.append(
                    f'<td style="padding:8px;text-align:center;background:{color};color:{text_color};'
                    f'font-family:monospace;font-size:11px;font-weight:bold;">{display_value}</td>'
                )
            
            rows.append(f'<tr>{"".join(cells)}</tr>')
        
        # Legend
        legend = '''
        <div style="display:flex;align-items:center;margin-top:10px;gap:20px;">
            <span style="color:#64748b;font-size:11px;">Legend:</span>
            <div style="display:flex;align-items:center;gap:5px;">
                <div style="width:20px;height:12px;background:rgb(60,180,60);border-radius:2px;"></div>
                <span style="color:#64748b;font-size:11px;">Decreases fraud</span>
            </div>
            <div style="display:flex;align-items:center;gap:5px;">
                <div style="width:20px;height:12px;background:#1e293b;border:1px solid #334155;border-radius:2px;"></div>
                <span style="color:#64748b;font-size:11px;">Neutral</span>
            </div>
            <div style="display:flex;align-items:center;gap:5px;">
                <div style="width:20px;height:12px;background:rgb(240,80,80);border-radius:2px;"></div>
                <span style="color:#64748b;font-size:11px;">Increases fraud</span>
            </div>
        </div>
        '''
        
        return f'''
        <div style="margin-bottom:25px;">
            <h4 style="color:#94a3b8;margin:0 0 10px 0;">🗺️ CATE Heatmap: Effect Heterogeneity</h4>
            <div style="overflow-x:auto;">
                <table style="border-collapse:collapse;background:#1e293b;border-radius:8px;overflow:hidden;">
                    <thead>
                        <tr style="background:#334155;">{"".join(header_cells)}</tr>
                    </thead>
                    <tbody>{"".join(rows)}</tbody>
                </table>
            </div>
            {legend}
        </div>
        '''
    
    def _generate_cate_bar_charts(self) -> str:
        """Generate CATE bar charts for top treatments."""
        if not self.cate_results:
            return ""
        
        # Group by treatment and find those with most heterogeneity
        by_treatment: dict[str, list[CATEResult]] = {}
        for r in self.cate_results:
            if r.treatment not in by_treatment:
                by_treatment[r.treatment] = []
            by_treatment[r.treatment].append(r)
        
        # Calculate heterogeneity (range of CATE values)
        heterogeneity = []
        for treatment, results in by_treatment.items():
            cates = [r.cate for r in results if not np.isnan(r.cate)]
            if len(cates) >= 2:
                het_range = max(cates) - min(cates)
                heterogeneity.append((treatment, het_range, results))
        
        # Sort by heterogeneity and take top 3
        heterogeneity.sort(key=lambda x: x[1], reverse=True)
        top_treatments = heterogeneity[:3]
        
        if not top_treatments:
            return ""
        
        charts = []
        for treatment, het_range, results in top_treatments:
            # Sort results by CATE value
            sorted_results = sorted(results, key=lambda r: r.cate, reverse=True)
            
            # Find max for scaling
            max_abs = max(abs(r.cate) for r in sorted_results) or 1
            
            bars = []
            for r in sorted_results:
                if np.isnan(r.cate):
                    continue
                
                # Calculate bar width (0-200px)
                width = int(abs(r.cate) / max_abs * 180)
                
                # Color based on sign
                if r.cate > 0:
                    color = "#ef4444"  # Red
                    direction = "right"
                    margin = "margin-left:100px;"
                else:
                    color = "#22c55e"  # Green
                    direction = "left"
                    margin = f"margin-left:{100 - width}px;"
                
                # Significance indicator
                sig_indicator = "●" if r.is_significant else "○"
                sig_color = "#22c55e" if r.is_significant else "#64748b"
                
                bars.append(f'''
                <div style="display:flex;align-items:center;margin:6px 0;">
                    <span style="width:140px;color:#e2e8f0;font-size:11px;text-align:right;padding-right:10px;">
                        {r.subgroup_name[:18]}
                    </span>
                    <div style="width:200px;position:relative;height:18px;">
                        <div style="position:absolute;left:99px;top:0;width:2px;height:18px;background:#334155;"></div>
                        <div style="position:absolute;{margin}width:{width}px;height:18px;background:{color};border-radius:2px;"></div>
                    </div>
                    <span style="width:70px;color:{color};font-family:monospace;font-size:11px;padding-left:10px;">
                        {r.cate:+.4f}
                    </span>
                    <span style="color:{sig_color};font-size:10px;">{sig_indicator}</span>
                </div>
                ''')
            
            # Calculate insight
            max_result = max(sorted_results, key=lambda r: r.cate)
            min_result = min(sorted_results, key=lambda r: r.cate)
            multiplier = abs(max_result.cate / min_result.cate) if min_result.cate != 0 else 0
            
            insight = ""
            if multiplier > 1.5:
                insight = f'''
                <div style="background:#334155;padding:8px 12px;border-radius:4px;margin-top:8px;border-left:3px solid #fbbf24;">
                    <span style="color:#fbbf24;font-size:11px;">💡 Insight:</span>
                    <span style="color:#e2e8f0;font-size:11px;">
                        Effect is <b>{multiplier:.1f}x stronger</b> for "{max_result.subgroup_name}" 
                        vs "{min_result.subgroup_name}"
                    </span>
                </div>
                '''
            
            display_treatment = treatment[:35] + "..." if len(treatment) > 35 else treatment
            
            charts.append(f'''
            <div style="background:#1e293b;padding:15px;border-radius:8px;margin-bottom:15px;">
                <h5 style="color:#fbbf24;margin:0 0 12px 0;font-size:13px;">
                    {display_treatment}
                    <span style="color:#64748b;font-weight:normal;font-size:11px;margin-left:10px;">
                        (heterogeneity: {het_range:.4f})
                    </span>
                </h5>
                <div style="display:flex;justify-content:center;margin-bottom:5px;">
                    <span style="color:#22c55e;font-size:10px;width:100px;text-align:right;">← Decreases fraud</span>
                    <span style="width:20px;"></span>
                    <span style="color:#ef4444;font-size:10px;width:100px;">Increases fraud →</span>
                </div>
                {"".join(bars)}
                {insight}
            </div>
            ''')
        
        return f'''
        <div style="margin-bottom:25px;">
            <h4 style="color:#94a3b8;margin:0 0 15px 0;">📊 CATE Comparison: Effect by Subgroup</h4>
            <p style="color:#64748b;font-size:12px;margin-bottom:15px;">
                Showing treatments with highest effect heterogeneity (● = significant, ○ = not significant)
            </p>
            {"".join(charts)}
        </div>
        '''
    
    def _generate_interpretation_guide(self) -> str:
        """Generate interpretation guide."""
        return '''
        <div style="background:#334155;padding:15px;border-radius:8px;margin-top:20px;">
            <h4 style="color:#94a3b8;margin:0 0 10px 0;">📖 How to Interpret</h4>
            <ul style="color:#94a3b8;font-size:12px;margin:0;padding-left:20px;">
                <li><b>ATE (β)</b>: Change in outcome per 1-unit change in treatment</li>
                <li><b style="color:#ef4444;">Positive</b>: Increases fraud probability | 
                    <b style="color:#22c55e;">Negative</b>: Decreases fraud probability</li>
                <li><b>*** p<0.001</b>, <b>** p<0.01</b>, <b>* p<0.05</b>: Statistical significance</li>
                <li><b>95% CI</b>: We're 95% confident the true effect is in this range</li>
            </ul>
        </div>
        '''
    
    def to_dict(self) -> dict[str, Any]:
        """Convert full report to dictionary."""
        return {
            "metadata": {
                "timestamp": self.timestamp,
                "outcome_variable": self.outcome_variable,
                "total_sample_size": self.total_sample_size,
                "n_ate_results": len(self.ate_results),
                "n_cate_results": len(self.cate_results),
            },
            "ate_results": [r.to_dict() for r in self.ate_results],
            "cate_results": [r.to_dict() for r in self.cate_results],
            "summary": {
                "significant_effects": [r.treatment for r in self.get_significant_effects()],
                "top_positive_effect": self._get_top_effect(positive=True),
                "top_negative_effect": self._get_top_effect(positive=False),
            },
        }
    
    def _get_top_effect(self, positive: bool) -> dict[str, Any] | None:
        """Get top positive or negative effect."""
        filtered = [r for r in self.ate_results if (r.ate > 0) == positive and r.is_significant]
        if not filtered:
            return None
        top = max(filtered, key=lambda x: abs(x.ate))
        return {"treatment": top.treatment, "ate": round(top.ate, 4)}
    
    def to_json(self, path: str | Path) -> None:
        """Save report to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"✓ Report saved: {path}")
    
    def to_html(self, path: str | Path) -> None:
        """
        Save report as standalone HTML file.
        
        Args:
            path: Output file path (e.g., "report.html")
        
        Example:
            report.to_html("causal_effects_report.html")
        """
        html_content = self._generate_html_report()
        
        full_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Causal Effect Estimation Report</title>
    <style>
        body {{
            background: #0f172a;
            padding: 20px;
            margin: 0;
            min-height: 100vh;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>'''
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(full_html)
        
        print(f"✓ HTML report saved: {path}")


# =============================================================================
# MAIN ESTIMATOR CLASS
# =============================================================================

class CausalEffectEstimator:
    """
    Causal Effect Estimator using regression adjustment.
    
    Estimates Average Treatment Effects (ATE) and Conditional Average
    Treatment Effects (CATE) based on an approved causal graph.
    
    Example:
        estimator = CausalEffectEstimator.from_files(
            approved_graph_path="approved_graph.json",
            data_path="fraud_data.csv",
        )
        report = estimator.estimate_all()
        report.display()
    """
    
    def __init__(
        self,
        approved_edges: list[CausalEdgeInfo],
        data: pd.DataFrame,
        outcome_variable: str = "is_fraud",
    ) -> None:
        """
        Initialize the estimator.
        
        Args:
            approved_edges: List of approved causal edges.
            data: DataFrame with all variables.
            outcome_variable: Name of the outcome variable.
        """
        self._edges = approved_edges
        self._data = data.copy()
        self._outcome = outcome_variable
        
        # Validate and filter edges
        self._validate_inputs()
        
        # Precompute treatments (after filtering)
        self._treatments = list({e.source for e in self._edges if e.target == outcome_variable})
        self._n_samples = len(data)
    
    @classmethod
    def from_files(
        cls,
        approved_graph_path: str | Path,
        data_path: str | Path,
        outcome_variable: str = "is_fraud",
    ) -> "CausalEffectEstimator":
        """
        Create estimator from files.
        
        Args:
            approved_graph_path: Path to approved_graph.json from review.
            data_path: Path to CSV data file.
            outcome_variable: Name of outcome variable.
        
        Returns:
            Configured CausalEffectEstimator instance.
        """
        # Load approved graph
        with open(approved_graph_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)
        
        # Parse edges
        edges = []
        for edge_dict in graph_data.get("approved_edges", []):
            edge = CausalEdgeInfo(
                source=edge_dict["source"],
                target=edge_dict["target"],
                approved_strength=edge_dict.get("approved_strength") or edge_dict.get("discovered_strength", 0.0),
                is_confounded=edge_dict.get("is_confounded", False),
                confounders=tuple(edge_dict.get("confounders", [])),
            )
            edges.append(edge)
        
        # Load data
        data = pd.read_csv(data_path)
        
        return cls(approved_edges=edges, data=data, outcome_variable=outcome_variable)
    
    @classmethod
    def from_dict(
        cls,
        approved_graph: dict[str, Any],
        data: pd.DataFrame,
        outcome_variable: str = "is_fraud",
    ) -> "CausalEffectEstimator":
        """
        Create estimator from dictionary and DataFrame.
        
        Args:
            approved_graph: Dictionary from CausalGraphReviewer.export_approved().
            data: DataFrame with variables.
            outcome_variable: Name of outcome variable.
        
        Returns:
            Configured CausalEffectEstimator instance.
        """
        edges = []
        for edge_dict in approved_graph.get("approved_edges", []):
            edge = CausalEdgeInfo(
                source=edge_dict["source"],
                target=edge_dict["target"],
                approved_strength=edge_dict.get("approved_strength") or edge_dict.get("discovered_strength", 0.0),
                is_confounded=edge_dict.get("is_confounded", False),
                confounders=tuple(edge_dict.get("confounders", [])),
            )
            edges.append(edge)
        
        return cls(approved_edges=edges, data=data, outcome_variable=outcome_variable)
    
    def _validate_inputs(self) -> None:
        """Validate inputs and filter edges with missing variables."""
        # Check outcome variable
        if self._outcome not in self._data.columns:
            available = ", ".join(self._data.columns[:5])
            raise ValueError(
                f"Outcome variable '{self._outcome}' not found in data. "
                f"Available columns: {available}..."
            )
        
        # Filter edges - keep only those with variables present in data
        valid_edges = []
        skipped = []
        
        for edge in self._edges:
            if edge.source not in self._data.columns:
                skipped.append(edge.source)
            elif edge.target not in self._data.columns:
                skipped.append(edge.target)
            else:
                valid_edges.append(edge)
        
        if skipped:
            print(f"⚠ Skipped edges with latent variables: {set(skipped)}")
        
        self._edges = valid_edges
    
    def estimate_ate(
        self,
        treatment: str,
        covariates: list[str] | None = None,
    ) -> ATEResult:
        """
        Estimate Average Treatment Effect using OLS regression.
        
        Uses regression adjustment: Y ~ T + covariates
        ATE is the coefficient on T.
        
        Args:
            treatment: Name of treatment variable.
            covariates: Optional list of control variables.
        
        Returns:
            ATEResult with estimate and statistics.
        """
        # Prepare data
        df = self._data[[treatment, self._outcome]].dropna()
        
        if covariates:
            valid_covs = [c for c in covariates if c in self._data.columns and c != treatment]
            if valid_covs:
                df = self._data[[treatment, self._outcome] + valid_covs].dropna()
        
        n = len(df)
        if n < 10:
            raise ValueError(f"Insufficient data for treatment '{treatment}': only {n} observations.")
        
        # Get treatment and outcome values
        X = df[treatment].values
        Y = df[self._outcome].values.astype(float)
        
        # Check for zero variance in treatment (would make regression undefined)
        x_var = float(np.var(X))
        if x_var < 1e-10:
            # Return a result indicating undefined effect
            return ATEResult(
                treatment=treatment,
                outcome=self._outcome,
                ate=0.0,
                std_error=float('inf'),
                ci_lower=float('-inf'),
                ci_upper=float('inf'),
                p_value=1.0,
                t_statistic=0.0,
                sample_size=n,
                r_squared=0.0,
                method="OLS (undefined - zero variance in treatment)",
            )
        
        # Build design matrix
        if covariates:
            valid_covs = [c for c in covariates if c in df.columns and c != treatment]
            if valid_covs:
                cov_matrix = df[valid_covs].values
                X_full = np.column_stack([np.ones(n), X, cov_matrix])
            else:
                X_full = np.column_stack([np.ones(n), X])
        else:
            X_full = np.column_stack([np.ones(n), X])
        
        # OLS estimation
        try:
            # Use numpy lstsq for numerical stability
            coeffs, residuals, rank, s = np.linalg.lstsq(X_full, Y, rcond=None)
            
            # Get treatment coefficient (index 1)
            ate = float(coeffs[1])
            
            # Calculate statistics
            Y_pred = X_full @ coeffs
            residuals = Y - Y_pred
            
            # Residual sum of squares
            rss = float(np.sum(residuals ** 2))
            
            # Total sum of squares
            tss = float(np.sum((Y - np.mean(Y)) ** 2))
            
            # R-squared
            r_squared = 1 - (rss / tss) if tss > 0 else 0.0
            
            # Degrees of freedom
            k = X_full.shape[1]
            dof = n - k
            
            # Standard error of coefficients
            if dof > 0 and rss > 0:
                mse = rss / dof
                try:
                    cov_matrix = mse * np.linalg.inv(X_full.T @ X_full)
                    std_error = float(np.sqrt(cov_matrix[1, 1]))
                except np.linalg.LinAlgError:
                    # Fallback: bootstrap or simple estimate
                    std_error = float(np.std(residuals) / np.sqrt(n))
            else:
                std_error = float(np.std(residuals) / np.sqrt(n))
            
            # T-statistic and p-value
            if std_error > 0:
                t_stat = ate / std_error
                p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), dof)))
            else:
                t_stat = 0.0
                p_value = 1.0
            
            # Confidence interval (95%)
            t_critical = stats.t.ppf(0.975, dof) if dof > 0 else 1.96
            ci_lower = ate - t_critical * std_error
            ci_upper = ate + t_critical * std_error
            
        except Exception as e:
            raise RuntimeError(f"OLS estimation failed for '{treatment}': {e}") from e
        
        return ATEResult(
            treatment=treatment,
            outcome=self._outcome,
            ate=ate,
            std_error=std_error,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            t_statistic=t_stat,
            sample_size=n,
            r_squared=r_squared,
            method="OLS Regression Adjustment",
        )
    
    def estimate_cate(
        self,
        treatment: str,
        subgroup: SubgroupDefinition,
    ) -> CATEResult:
        """
        Estimate Conditional Average Treatment Effect for a subgroup.
        
        Args:
            treatment: Name of treatment variable.
            subgroup: Subgroup definition with condition.
        
        Returns:
            CATEResult with subgroup-specific estimate.
        """
        # Get subgroup mask
        mask = subgroup.condition(self._data)
        df_subgroup = self._data[mask]
        
        n_subgroup = len(df_subgroup)
        n_total = len(self._data)
        
        if n_subgroup < 10:
            # Return null result for small subgroups
            return CATEResult(
                treatment=treatment,
                outcome=self._outcome,
                subgroup_name=subgroup.name,
                subgroup_description=subgroup.description,
                cate=np.nan,
                std_error=np.nan,
                ci_lower=np.nan,
                ci_upper=np.nan,
                p_value=1.0,
                sample_size=n_subgroup,
                subgroup_proportion=n_subgroup / n_total if n_total > 0 else 0,
            )
        
        # Estimate ATE on subgroup
        df = df_subgroup[[treatment, self._outcome]].dropna()
        n = len(df)
        
        X = np.column_stack([np.ones(n), df[treatment].values])
        Y = df[self._outcome].values.astype(float)
        
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            cate = float(coeffs[1])
            
            # Statistics
            Y_pred = X @ coeffs
            residuals = Y - Y_pred
            rss = float(np.sum(residuals ** 2))
            dof = n - 2
            
            if dof > 0 and rss > 0:
                mse = rss / dof
                try:
                    cov_matrix = mse * np.linalg.inv(X.T @ X)
                    std_error = float(np.sqrt(cov_matrix[1, 1]))
                except np.linalg.LinAlgError:
                    std_error = float(np.std(residuals) / np.sqrt(n))
            else:
                std_error = float(np.std(residuals) / np.sqrt(n))
            
            if std_error > 0:
                t_stat = cate / std_error
                p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), max(dof, 1))))
            else:
                p_value = 1.0
            
            t_critical = stats.t.ppf(0.975, max(dof, 1))
            ci_lower = cate - t_critical * std_error
            ci_upper = cate + t_critical * std_error
            
        except Exception:
            cate = np.nan
            std_error = np.nan
            ci_lower = np.nan
            ci_upper = np.nan
            p_value = 1.0
        
        return CATEResult(
            treatment=treatment,
            outcome=self._outcome,
            subgroup_name=subgroup.name,
            subgroup_description=subgroup.description,
            cate=cate,
            std_error=std_error,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
            sample_size=n_subgroup,
            subgroup_proportion=n_subgroup / n_total,
        )
    
    def estimate_all(
        self,
        subgroups: list[SubgroupDefinition] | None = None,
    ) -> CausalEffectReport:
        """
        Estimate all causal effects from approved graph.
        
        Args:
            subgroups: Optional list of subgroup definitions for CATE.
                      If None, uses default fraud-relevant subgroups.
        
        Returns:
            CausalEffectReport with all results.
        """
        # Default subgroups for fraud detection
        if subgroups is None:
            subgroups = self._get_default_subgroups()
        
        # Estimate ATE for each treatment
        ate_results: list[ATEResult] = []
        for treatment in self._treatments:
            try:
                result = self.estimate_ate(treatment)
                ate_results.append(result)
            except Exception as e:
                print(f"⚠ Could not estimate ATE for '{treatment}': {e}")
        
        # Estimate CATE for each treatment × subgroup
        cate_results: list[CATEResult] = []
        for treatment in self._treatments:
            for subgroup in subgroups:
                try:
                    result = self.estimate_cate(treatment, subgroup)
                    if not np.isnan(result.cate):
                        cate_results.append(result)
                except Exception:
                    pass  # Skip failed subgroup estimates
        
        return CausalEffectReport(
            ate_results=ate_results,
            cate_results=cate_results,
            outcome_variable=self._outcome,
            total_sample_size=self._n_samples,
        )
    
    def _get_default_subgroups(self) -> list[SubgroupDefinition]:
        """Get default subgroups relevant for fraud detection."""
        subgroups = []
        
        # Account age subgroups
        if "account_age_days" in self._data.columns:
            median_age = self._data["account_age_days"].median()
            
            subgroups.append(SubgroupDefinition(
                name="New Accounts",
                description=f"Account age < {median_age:.0f} days",
                condition=lambda df, m=median_age: df["account_age_days"] < m,
            ))
            subgroups.append(SubgroupDefinition(
                name="Established Accounts",
                description=f"Account age ≥ {median_age:.0f} days",
                condition=lambda df, m=median_age: df["account_age_days"] >= m,
            ))
        
        # Transaction amount subgroups
        if "transaction_amount" in self._data.columns:
            q75 = self._data["transaction_amount"].quantile(0.75)
            
            subgroups.append(SubgroupDefinition(
                name="High-Value Transactions",
                description=f"Amount > {q75:,.0f} (top 25%)",
                condition=lambda df, q=q75: df["transaction_amount"] > q,
            ))
            subgroups.append(SubgroupDefinition(
                name="Standard Transactions",
                description=f"Amount ≤ {q75:,.0f}",
                condition=lambda df, q=q75: df["transaction_amount"] <= q,
            ))
        
        # Foreign transaction subgroups
        if "is_foreign_transaction" in self._data.columns:
            subgroups.append(SubgroupDefinition(
                name="Foreign Transactions",
                description="Cross-border transactions",
                condition=lambda df: df["is_foreign_transaction"] == 1,
            ))
            subgroups.append(SubgroupDefinition(
                name="Domestic Transactions",
                description="Local transactions",
                condition=lambda df: df["is_foreign_transaction"] == 0,
            ))
        
        return subgroups


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def estimate_causal_effects(
    approved_graph_path: str = "approved_graph.json",
    data_path: str = "fraud_data.csv",
    outcome: str = "is_fraud",
    display_report: bool = True,
) -> CausalEffectReport:
    """
    Quick-start function to estimate all causal effects.
    
    Args:
        approved_graph_path: Path to approved graph JSON.
        data_path: Path to data CSV.
        outcome: Outcome variable name.
        display_report: Whether to display HTML report.
    
    Returns:
        CausalEffectReport with all results.
    
    Example:
        report = estimate_causal_effects()
        report.to_json("effects.json")
    """
    estimator = CausalEffectEstimator.from_files(
        approved_graph_path=approved_graph_path,
        data_path=data_path,
        outcome_variable=outcome,
    )
    
    report = estimator.estimate_all()
    
    if display_report:
        report.display()
    
    return report


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("Causal Effect Estimator")
    print("=" * 40)
    print()
    print("Usage in Python:")
    print("  from causal_effect_estimator import estimate_causal_effects")
    print("  report = estimate_causal_effects()")
    print()
    print("Or step by step:")
    print("  from causal_effect_estimator import CausalEffectEstimator")
    print("  estimator = CausalEffectEstimator.from_files(")
    print('      approved_graph_path="approved_graph.json",')
    print('      data_path="fraud_data.csv",')
    print("  )")
    print("  report = estimator.estimate_all()")
    print("  report.display()")
