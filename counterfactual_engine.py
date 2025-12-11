"""
counterfactual_engine.py

Counterfactual Reasoning Engine - odpowiada na pytania "Co by było gdyby?"
dla pojedynczych transakcji w kontekście fraud detection.

Integracja z istniejącym projektem:
    from counterfactual_engine import CounterfactualEngine
    
    engine = CounterfactualEngine.from_files(
        approved_graph_path="approved_graph.json",
        effects_report_path="causal_effects_report.json",
    )
    
    # Analiza pojedynczej transakcji
    result = engine.analyze(transaction)
    result.display()
    
    # "Co zmienić żeby P(fraud) spadło poniżej 20%?"
    recommendation = engine.recommend(transaction, target_probability=0.20)

Wymagania:
    pip install pandas numpy

Author: Causal AI Engine
Version: 1.0.0
PEP: 8, 257, 484, 585, 604
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from IPython.display import display, HTML


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True, slots=True)
class CausalFactor:
    """Single causal factor with its effect."""
    name: str
    ate: float
    current_value: float
    contribution: float  # ate * current_value
    unit: str = ""
    
    @property
    def direction(self) -> str:
        """Direction of effect."""
        return "increases" if self.ate > 0 else "decreases"
    
    @property
    def direction_pl(self) -> str:
        """Direction of effect in Polish."""
        return "zwiększa" if self.ate > 0 else "zmniejsza"


@dataclass(frozen=True, slots=True)
class CounterfactualScenario:
    """Single counterfactual scenario: what if X was different?"""
    factor_name: str
    original_value: float
    counterfactual_value: float
    original_probability: float
    counterfactual_probability: float
    probability_change: float
    unit: str = ""
    
    @property
    def change_description(self) -> str:
        """Human-readable description of the change."""
        if self.counterfactual_value > self.original_value:
            direction = "increased"
        else:
            direction = "decreased"
        return f"{self.factor_name} {direction} from {self.original_value:.2f} to {self.counterfactual_value:.2f}"
    
    @property
    def effect_description(self) -> str:
        """Human-readable description of the effect."""
        change_pp = self.probability_change * 100
        if change_pp > 0:
            return f"P(fraud) would increase by {change_pp:.1f}pp"
        else:
            return f"P(fraud) would decrease by {abs(change_pp):.1f}pp"


@dataclass(frozen=True, slots=True)
class Recommendation:
    """Recommendation for reducing fraud probability."""
    factor_name: str
    current_value: float
    recommended_value: float
    probability_reduction: float
    feasibility: str  # "easy", "medium", "hard", "impossible"
    explanation: str
    
    @property
    def is_feasible(self) -> bool:
        return self.feasibility in ("easy", "medium")


@dataclass
class CounterfactualAnalysis:
    """
    Complete counterfactual analysis for a single transaction.
    
    Contains:
    - Current state analysis
    - Factor contributions
    - Counterfactual scenarios
    - Recommendations
    """
    transaction_id: str
    transaction_data: dict[str, Any]
    baseline_probability: float
    factors: list[CausalFactor]
    scenarios: list[CounterfactualScenario]
    recommendations: list[Recommendation]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def get_top_contributors(self, n: int = 3) -> list[CausalFactor]:
        """Get top N factors contributing to fraud risk."""
        # Sort by absolute contribution, but prioritize positive (increasing fraud)
        return sorted(
            self.factors,
            key=lambda f: (-f.contribution if f.contribution > 0 else abs(f.contribution) * 0.5),
            reverse=True,
        )[:n]
    
    def get_actionable_recommendations(self) -> list[Recommendation]:
        """Get only feasible recommendations."""
        return [r for r in self.recommendations if r.is_feasible]
    
    def display(self) -> None:
        """Display formatted analysis in Jupyter."""
        html = self._generate_html_report()
        display(HTML(html))
    
    def _generate_html_report(self) -> str:
        """Generate HTML report."""
        prob_color = "#ef4444" if self.baseline_probability > 0.5 else "#fbbf24" if self.baseline_probability > 0.3 else "#22c55e"
        
        html_parts = [
            '<div style="font-family:Arial,sans-serif;max-width:900px;">',
            
            # Header
            f'''
            <div style="background:linear-gradient(135deg,#1e3a5f,#0f172a);padding:20px;border-radius:8px;margin-bottom:20px;">
                <h2 style="margin:0;color:#60a5fa;">🔮 Counterfactual Analysis</h2>
                <p style="margin:8px 0 0 0;color:#94a3b8;">
                    Transaction: <b style="color:white;">{self.transaction_id}</b> |
                    P(fraud): <b style="color:{prob_color};">{self.baseline_probability:.1%}</b>
                </p>
            </div>
            ''',
            
            # Current state
            '<h3 style="color:#e2e8f0;margin-bottom:10px;">📊 Current State Analysis</h3>',
            '<p style="color:#94a3b8;font-size:13px;margin-bottom:15px;">'
            'How each factor contributes to the fraud probability:</p>',
            self._generate_contribution_chart(),
            
            # Counterfactual scenarios
            '<h3 style="color:#e2e8f0;margin:25px 0 10px 0;">🔄 "What If?" Scenarios</h3>',
            '<p style="color:#94a3b8;font-size:13px;margin-bottom:15px;">'
            'How would the fraud probability change under different conditions?</p>',
            self._generate_scenarios_table(),
            
            # Recommendations
            '<h3 style="color:#e2e8f0;margin:25px 0 10px 0;">💡 Recommendations</h3>',
            '<p style="color:#94a3b8;font-size:13px;margin-bottom:15px;">'
            'Actions that could reduce fraud risk:</p>',
            self._generate_recommendations(),
            
            # Explanation box
            self._generate_explanation_box(),
            
            '</div>',
        ]
        
        return '\n'.join(html_parts)
    
    def _generate_contribution_chart(self) -> str:
        """Generate factor contribution chart."""
        if not self.factors:
            return '<p style="color:#64748b;">No factors available.</p>'
        
        max_abs = max(abs(f.contribution) for f in self.factors) or 1
        
        bars = []
        for f in sorted(self.factors, key=lambda x: x.contribution, reverse=True):
            width = int(abs(f.contribution) / max_abs * 150)
            
            if f.contribution > 0:
                color = "#ef4444"
                direction = "→ zwiększa ryzyko"
                bar_style = f"margin-left:150px;width:{width}px;"
            else:
                color = "#22c55e"
                direction = "← zmniejsza ryzyko"
                bar_style = f"margin-left:{150-width}px;width:{width}px;"
            
            contribution_pp = f.contribution * 100
            
            bars.append(f'''
            <div style="display:flex;align-items:center;margin:8px 0;">
                <span style="width:200px;color:#e2e8f0;font-size:12px;text-align:right;padding-right:10px;">
                    {f.name}<br>
                    <span style="color:#64748b;font-size:10px;">= {f.current_value:.2f}</span>
                </span>
                <div style="width:300px;position:relative;height:24px;background:#1e293b;border-radius:4px;">
                    <div style="position:absolute;left:149px;top:0;width:2px;height:24px;background:#334155;"></div>
                    <div style="position:absolute;{bar_style}height:24px;background:{color};border-radius:4px;"></div>
                </div>
                <span style="width:80px;color:{color};font-family:monospace;font-size:11px;padding-left:10px;">
                    {contribution_pp:+.2f}pp
                </span>
                <span style="color:#64748b;font-size:10px;width:120px;">{direction}</span>
            </div>
            ''')
        
        return f'''
        <div style="background:#0f172a;padding:15px;border-radius:8px;">
            <div style="display:flex;justify-content:center;margin-bottom:10px;">
                <span style="color:#22c55e;font-size:10px;width:150px;text-align:right;">← Zmniejsza P(fraud)</span>
                <span style="width:20px;"></span>
                <span style="color:#ef4444;font-size:10px;width:150px;">Zwiększa P(fraud) →</span>
            </div>
            {"".join(bars)}
        </div>
        '''
    
    def _generate_scenarios_table(self) -> str:
        """Generate counterfactual scenarios table."""
        if not self.scenarios:
            return '<p style="color:#64748b;">No scenarios available.</p>'
        
        rows = []
        for s in self.scenarios:
            prob_change_pp = s.probability_change * 100
            new_prob_pct = s.counterfactual_probability * 100
            
            if prob_change_pp < 0:
                color = "#22c55e"
                arrow = "↓"
            else:
                color = "#ef4444"
                arrow = "↑"
            
            rows.append(f'''
            <tr style="border-bottom:1px solid #334155;">
                <td style="padding:10px;color:#fbbf24;">{s.factor_name}</td>
                <td style="padding:10px;color:#94a3b8;font-family:monospace;">
                    {s.original_value:.2f} → <b style="color:#60a5fa;">{s.counterfactual_value:.2f}</b>
                    <span style="color:#64748b;font-size:10px;"> (p25/p75)</span>
                </td>
                <td style="padding:10px;color:{color};font-weight:bold;font-family:monospace;">
                    {arrow} {prob_change_pp:+.1f}pp
                </td>
                <td style="padding:10px;color:#e2e8f0;font-family:monospace;">
                    {new_prob_pct:.1f}%
                </td>
            </tr>
            ''')
        
        return f'''
        <table style="width:100%;border-collapse:collapse;background:#1e293b;border-radius:8px;overflow:hidden;">
            <thead>
                <tr style="background:#334155;">
                    <th style="padding:12px;text-align:left;color:#94a3b8;font-weight:normal;">Czynnik</th>
                    <th style="padding:12px;text-align:left;color:#94a3b8;font-weight:normal;">Aktualna → Percentyl</th>
                    <th style="padding:12px;text-align:left;color:#94a3b8;font-weight:normal;">Zmiana P(fraud)</th>
                    <th style="padding:12px;text-align:left;color:#94a3b8;font-weight:normal;">Nowe P(fraud)</th>
                </tr>
            </thead>
            <tbody>{"".join(rows)}</tbody>
        </table>
        <p style="color:#64748b;font-size:11px;margin-top:8px;">
            💡 Wartości counterfactual pochodzą z rozkładu danych: p25 = 25. percentyl, p75 = 75. percentyl
        </p>
        '''
    
    def _generate_recommendations(self) -> str:
        """Generate recommendations section."""
        if not self.recommendations:
            return '<p style="color:#64748b;">No recommendations available.</p>'
        
        cards = []
        for i, r in enumerate(self.recommendations[:3], 1):
            feasibility_colors = {
                "easy": "#22c55e",
                "medium": "#fbbf24",
                "hard": "#f97316",
                "impossible": "#ef4444",
            }
            feas_color = feasibility_colors.get(r.feasibility, "#64748b")
            reduction_pp = r.probability_reduction * 100
            
            cards.append(f'''
            <div style="background:#1e293b;padding:15px;border-radius:8px;margin-bottom:10px;border-left:4px solid {feas_color};">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                    <span style="color:#e2e8f0;font-weight:bold;">#{i} {r.factor_name}</span>
                    <span style="background:{feas_color};color:#0f172a;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:bold;">
                        {r.feasibility.upper()}
                    </span>
                </div>
                <p style="color:#94a3b8;margin:0 0 8px 0;font-size:13px;">
                    Zmień z <b style="color:#fbbf24;">{r.current_value:.2f}</b> 
                    na <b style="color:#22c55e;">{r.recommended_value:.2f}</b>
                </p>
                <p style="color:#22c55e;margin:0;font-size:13px;">
                    → P(fraud) spadnie o <b>{reduction_pp:.1f}pp</b>
                </p>
                <p style="color:#64748b;margin:8px 0 0 0;font-size:11px;font-style:italic;">
                    {r.explanation}
                </p>
            </div>
            ''')
        
        return ''.join(cards)
    
    def _generate_explanation_box(self) -> str:
        """Generate explanation box."""
        top_contributors = self.get_top_contributors(3)
        
        reasons = []
        for f in top_contributors:
            if f.contribution > 0.01:
                reasons.append(f"• {f.name} = {f.current_value:.2f} (dodaje {f.contribution*100:.1f}pp)")
        
        if not reasons:
            reasons = ["• Brak dominujących czynników ryzyka"]
        
        return f'''
        <div style="background:#334155;padding:15px;border-radius:8px;margin-top:20px;">
            <h4 style="color:#60a5fa;margin:0 0 10px 0;">📝 Podsumowanie dla analityka</h4>
            <p style="color:#e2e8f0;margin:0;font-size:13px;">
                Transakcja ma <b>P(fraud) = {self.baseline_probability:.1%}</b>
            </p>
            <p style="color:#94a3b8;margin:10px 0 0 0;font-size:12px;">
                Główne czynniki ryzyka:<br>
                {"<br>".join(reasons)}
            </p>
        </div>
        '''
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "transaction_id": self.transaction_id,
            "transaction_data": self.transaction_data,
            "baseline_probability": round(self.baseline_probability, 4),
            "timestamp": self.timestamp,
            "factors": [
                {
                    "name": f.name,
                    "ate": round(f.ate, 6),
                    "current_value": round(f.current_value, 4),
                    "contribution": round(f.contribution, 6),
                }
                for f in self.factors
            ],
            "scenarios": [
                {
                    "factor": s.factor_name,
                    "original_value": round(s.original_value, 4),
                    "counterfactual_value": round(s.counterfactual_value, 4),
                    "original_probability": round(s.original_probability, 4),
                    "counterfactual_probability": round(s.counterfactual_probability, 4),
                    "probability_change": round(s.probability_change, 4),
                }
                for s in self.scenarios
            ],
            "recommendations": [
                {
                    "factor": r.factor_name,
                    "current_value": round(r.current_value, 4),
                    "recommended_value": round(r.recommended_value, 4),
                    "probability_reduction": round(r.probability_reduction, 4),
                    "feasibility": r.feasibility,
                    "explanation": r.explanation,
                }
                for r in self.recommendations
            ],
        }
    
    def to_json(self, path: str | Path) -> None:
        """Save analysis to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"✓ Analysis saved: {path}")


# =============================================================================
# MAIN ENGINE CLASS
# =============================================================================

class CounterfactualEngine:
    """
    Counterfactual Reasoning Engine for fraud detection.
    
    Answers questions like:
    - "Why was this transaction flagged?"
    - "What would happen if the amount was lower?"
    - "What should change to reduce fraud probability?"
    
    Example:
        engine = CounterfactualEngine.from_files(
            approved_graph_path="approved_graph.json",
            effects_report_path="causal_effects_report.json",
        )
        
        transaction = {
            "transaction_amount": 15000,
            "merchant_risk_score": 0.8,
            "transaction_velocity_24h": 6,
            ...
        }
        
        result = engine.analyze(transaction)
        result.display()
    """
    
    # Factor metadata for realistic scenarios
    # NOTE: actual counterfactual values are computed from DATA percentiles
    FACTOR_METADATA: dict[str, dict[str, Any]] = {
        "transaction_amount": {
            "unit": "PLN",
            "feasibility": "easy",
            "explanation": "Klient może podzielić transakcję na mniejsze",
            "counterfactual_strategy": "percentile",  # use data percentiles
        },
        "merchant_risk_score": {
            "unit": "score",
            "feasibility": "hard",
            "explanation": "Wymaga zmiany sprzedawcy na bardziej zaufanego",
            "counterfactual_strategy": "percentile",
        },
        "transaction_velocity_24h": {
            "unit": "tx/24h",
            "feasibility": "medium",
            "explanation": "Rozłożenie transakcji w czasie",
            "counterfactual_strategy": "percentile",
        },
        "account_age_days": {
            "unit": "days",
            "feasibility": "impossible",
            "explanation": "Wiek konta nie może być zmieniony",
            "counterfactual_strategy": "percentile",
        },
        "is_foreign_transaction": {
            "unit": "bool",
            "feasibility": "medium",
            "explanation": "Użycie krajowego kanału płatności",
            "counterfactual_strategy": "flip",  # binary: just flip 0↔1
        },
        "device_fingerprint_age_days": {
            "unit": "days",
            "feasibility": "impossible",
            "explanation": "Wiek urządzenia nie może być zmieniony",
            "counterfactual_strategy": "percentile",
        },
    }
    
    # Percentiles for counterfactual scenarios (computed from data)
    _data_percentiles: dict[str, dict[str, float]] | None = None
    
    def __init__(
        self,
        ate_effects: dict[str, float],
        baseline_fraud_rate: float = 0.05,
        data: pd.DataFrame | None = None,
    ) -> None:
        """
        Initialize the engine.
        
        Args:
            ate_effects: Dictionary mapping factor names to ATE values.
            baseline_fraud_rate: Baseline fraud probability (intercept).
            data: Optional DataFrame for computing realistic counterfactual values.
        """
        self._ate = ate_effects
        self._baseline = baseline_fraud_rate
        self._factors = list(ate_effects.keys())
        self._data = data
        
        # Compute percentiles from data for realistic counterfactuals
        self._data_percentiles: dict[str, dict[str, float]] = {}
        if data is not None:
            self._compute_percentiles(data)
    
    def _compute_percentiles(self, data: pd.DataFrame) -> None:
        """
        Compute percentiles from data for realistic counterfactual values.
        
        Stores p10, p25, p50, p75, p90 for each factor.
        """
        percentiles = [10, 25, 50, 75, 90]
        
        for factor in self._factors:
            if factor in data.columns:
                values = data[factor].dropna()
                if len(values) > 0:
                    self._data_percentiles[factor] = {
                        f"p{p}": float(np.percentile(values, p))
                        for p in percentiles
                    }
                    # Add min/max for reference
                    self._data_percentiles[factor]["min"] = float(values.min())
                    self._data_percentiles[factor]["max"] = float(values.max())
                    self._data_percentiles[factor]["mean"] = float(values.mean())
    
    @classmethod
    def from_files(
        cls,
        approved_graph_path: str | Path | None = None,
        effects_report_path: str | Path = "causal_effects_report.json",
        data_path: str | Path | None = None,
    ) -> "CounterfactualEngine":
        """
        Create engine from existing project files.
        
        Args:
            approved_graph_path: Path to approved_graph.json (optional).
            effects_report_path: Path to causal_effects_report.json.
            data_path: Path to data CSV for baseline estimation and percentiles.
        
        Returns:
            Configured CounterfactualEngine instance.
        """
        # Load ATE effects
        with open(effects_report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        
        ate_effects = {
            r["treatment"]: r["ate"]
            for r in report.get("ate_results", [])
        }
        
        # Load data for baseline and percentiles
        baseline = 0.05
        data = None
        if data_path and Path(data_path).exists():
            data = pd.read_csv(data_path)
            if "is_fraud" in data.columns:
                baseline = float(data["is_fraud"].mean())
        
        return cls(ate_effects=ate_effects, baseline_fraud_rate=baseline, data=data)
    
    def predict_probability(
        self,
        transaction: dict[str, Any],
    ) -> float:
        """
        Predict fraud probability for a transaction.
        
        Uses linear model: P(fraud) = baseline + Σ(ATE_i × X_i)
        
        Args:
            transaction: Transaction features dictionary.
        
        Returns:
            Predicted fraud probability (clipped to [0, 1]).
        """
        prob = self._baseline
        
        for factor, ate in self._ate.items():
            if factor in transaction:
                value = float(transaction[factor])
                prob += ate * value
        
        return float(np.clip(prob, 0, 1))
    
    def analyze(
        self,
        transaction: dict[str, Any],
        transaction_id: str | None = None,
    ) -> CounterfactualAnalysis:
        """
        Perform full counterfactual analysis for a transaction.
        
        Args:
            transaction: Transaction features dictionary.
            transaction_id: Optional ID for the transaction.
        
        Returns:
            CounterfactualAnalysis with factors, scenarios, and recommendations.
        """
        if transaction_id is None:
            transaction_id = f"TX_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        baseline_prob = self.predict_probability(transaction)
        
        # Calculate factor contributions
        factors = self._calculate_contributions(transaction)
        
        # Generate counterfactual scenarios
        scenarios = self._generate_scenarios(transaction, baseline_prob)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(transaction, baseline_prob)
        
        return CounterfactualAnalysis(
            transaction_id=transaction_id,
            transaction_data=dict(transaction),
            baseline_probability=baseline_prob,
            factors=factors,
            scenarios=scenarios,
            recommendations=recommendations,
        )
    
    def counterfactual(
        self,
        transaction: dict[str, Any],
        changes: dict[str, float],
    ) -> tuple[float, float]:
        """
        Calculate counterfactual probability under specific changes.
        
        Args:
            transaction: Original transaction.
            changes: Dictionary of {factor: new_value}.
        
        Returns:
            Tuple of (new_probability, probability_change).
        """
        # Calculate raw probabilities (before clipping) for accurate change
        original_raw = self._baseline
        for factor, ate in self._ate.items():
            if factor in transaction:
                original_raw += ate * float(transaction[factor])
        
        # Apply changes
        modified = dict(transaction)
        modified.update(changes)
        
        new_raw = self._baseline
        for factor, ate in self._ate.items():
            if factor in modified:
                new_raw += ate * float(modified[factor])
        
        # Calculate change on raw values, then clip for display
        prob_change = new_raw - original_raw
        new_prob = float(np.clip(new_raw, 0, 1))
        
        return new_prob, prob_change
    
    def recommend(
        self,
        transaction: dict[str, Any],
        target_probability: float = 0.20,
        max_recommendations: int = 3,
    ) -> list[Recommendation]:
        """
        Generate recommendations to achieve target fraud probability.
        
        Args:
            transaction: Transaction features dictionary.
            target_probability: Target P(fraud) to achieve.
            max_recommendations: Maximum number of recommendations.
        
        Returns:
            List of actionable recommendations.
        """
        current_prob = self.predict_probability(transaction)
        
        if current_prob <= target_probability:
            return []  # Already below target
        
        recommendations = self._generate_recommendations(
            transaction,
            current_prob,
            target_probability,
        )
        
        # Filter to feasible and sort by effectiveness
        feasible = [r for r in recommendations if r.feasibility in ("easy", "medium")]
        feasible.sort(key=lambda r: r.probability_reduction, reverse=True)
        
        return feasible[:max_recommendations]
    
    def _calculate_contributions(
        self,
        transaction: dict[str, Any],
    ) -> list[CausalFactor]:
        """Calculate contribution of each factor."""
        factors = []
        
        for factor, ate in self._ate.items():
            if factor in transaction:
                value = float(transaction[factor])
                contribution = ate * value
                
                metadata = self.FACTOR_METADATA.get(factor, {})
                unit = metadata.get("unit", "")
                
                factors.append(CausalFactor(
                    name=factor,
                    ate=ate,
                    current_value=value,
                    contribution=contribution,
                    unit=unit,
                ))
        
        return factors
    
    def _generate_scenarios(
        self,
        transaction: dict[str, Any],
        baseline_prob: float,
    ) -> list[CounterfactualScenario]:
        """Generate counterfactual scenarios for each factor."""
        scenarios = []
        
        for factor, ate in self._ate.items():
            if factor not in transaction:
                continue
            
            current_value = float(transaction[factor])
            metadata = self.FACTOR_METADATA.get(factor, {})
            
            # Generate meaningful counterfactual value
            cf_value = self._get_counterfactual_value(factor, current_value, metadata)
            
            if cf_value is None or cf_value == current_value:
                continue
            
            # Calculate new probability
            value_change = cf_value - current_value
            prob_change = ate * value_change
            new_prob = float(np.clip(baseline_prob + prob_change, 0, 1))
            
            scenarios.append(CounterfactualScenario(
                factor_name=factor,
                original_value=current_value,
                counterfactual_value=cf_value,
                original_probability=baseline_prob,
                counterfactual_probability=new_prob,
                probability_change=new_prob - baseline_prob,
                unit=metadata.get("unit", ""),
            ))
        
        # Sort by absolute probability change
        scenarios.sort(key=lambda s: abs(s.probability_change), reverse=True)
        
        return scenarios
    
    def _get_counterfactual_value(
        self,
        factor: str,
        current_value: float,
        metadata: dict[str, Any],
    ) -> float | None:
        """
        Get meaningful counterfactual value for a factor based on DATA.
        
        Strategy:
        - For factors that INCREASE fraud (positive ATE):
          → counterfactual = lower percentile (p25) to reduce risk
        - For factors that DECREASE fraud (negative ATE):
          → counterfactual = higher percentile (p75) to reduce risk
        - For binary factors: flip 0↔1
        
        This ensures counterfactuals are REALISTIC (from actual data distribution).
        """
        strategy = metadata.get("counterfactual_strategy", "percentile")
        
        # Binary variables: just flip
        if strategy == "flip" or factor == "is_foreign_transaction":
            return 0.0 if current_value > 0.5 else 1.0
        
        # Percentile-based counterfactuals
        if factor not in self._data_percentiles:
            return None  # No data available
        
        percentiles = self._data_percentiles[factor]
        ate = self._ate.get(factor, 0)
        
        # Determine which percentile to use based on ATE direction
        if ate > 0:
            # Positive ATE = factor increases fraud
            # → Show counterfactual with LOWER value (p25) to reduce fraud
            target_percentile = "p25"
        else:
            # Negative ATE = factor decreases fraud (e.g., account_age)
            # → Show counterfactual with HIGHER value (p75) to reduce fraud
            target_percentile = "p75"
        
        cf_value = percentiles.get(target_percentile)
        
        if cf_value is None:
            return None
        
        # Don't return counterfactual if current value is already better
        if ate > 0 and current_value <= cf_value:
            # Current is already low, show even lower (p10)
            cf_value = percentiles.get("p10", cf_value)
        elif ate < 0 and current_value >= cf_value:
            # Current is already high, show even higher (p90)
            cf_value = percentiles.get("p90", cf_value)
        
        # Skip if values are essentially the same
        if abs(cf_value - current_value) < 0.001:
            return None
        
        return float(cf_value)
    
    def _generate_recommendations(
        self,
        transaction: dict[str, Any],
        current_prob: float,
        target_prob: float | None = None,
    ) -> list[Recommendation]:
        """
        Generate recommendations for reducing fraud probability.
        
        Uses DATA PERCENTILES to suggest realistic target values.
        """
        recommendations = []
        
        for factor, ate in self._ate.items():
            if factor not in transaction:
                continue
            
            current_value = float(transaction[factor])
            metadata = self.FACTOR_METADATA.get(factor, {})
            feasibility = metadata.get("feasibility", "medium")
            explanation = metadata.get("explanation", "")
            
            # Skip factors that can't reduce fraud probability
            # (impossible to change OR wrong direction)
            
            # Get target value from data percentiles
            if factor not in self._data_percentiles:
                continue
            
            percentiles = self._data_percentiles[factor]
            
            if ate > 0:
                # Positive ATE → need to DECREASE value to reduce fraud
                # Recommend going to p25 (lower quartile)
                recommended_value = percentiles.get("p25", current_value)
                
                # Skip if already at or below target
                if current_value <= recommended_value:
                    continue
                    
            else:
                # Negative ATE → need to INCREASE value to reduce fraud
                # But for things like account_age, user can't change it
                # So we skip or mark as impossible
                recommended_value = percentiles.get("p75", current_value)
                
                if current_value >= recommended_value:
                    continue
            
            # Handle binary separately
            if factor == "is_foreign_transaction":
                if current_value > 0.5:  # Currently foreign
                    recommended_value = 0.0
                else:
                    continue  # Already domestic
            
            # Calculate reduction
            value_change = recommended_value - current_value
            prob_reduction = -ate * value_change  # negative because we want reduction
            
            if prob_reduction <= 0.001:  # Less than 0.1pp reduction
                continue
            
            recommendations.append(Recommendation(
                factor_name=factor,
                current_value=current_value,
                recommended_value=recommended_value,
                probability_reduction=prob_reduction,
                feasibility=feasibility,
                explanation=explanation,
            ))
        
        # Sort by probability reduction (descending)
        recommendations.sort(key=lambda r: r.probability_reduction, reverse=True)
        
        return recommendations


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_transaction(
    transaction: dict[str, Any],
    effects_report_path: str = "causal_effects_report.json",
    data_path: str | None = "fraud_data.csv",
    display_result: bool = True,
) -> CounterfactualAnalysis:
    """
    Quick-start function to analyze a single transaction.
    
    Args:
        transaction: Transaction features dictionary.
        effects_report_path: Path to causal effects report.
        data_path: Path to data CSV for baseline estimation.
        display_result: Whether to display HTML report.
    
    Returns:
        CounterfactualAnalysis with full analysis.
    
    Example:
        result = analyze_transaction({
            "transaction_amount": 25000,
            "merchant_risk_score": 0.75,
            "transaction_velocity_24h": 5,
            "account_age_days": 120,
            "is_foreign_transaction": 1,
        })
    """
    engine = CounterfactualEngine.from_files(
        effects_report_path=effects_report_path,
        data_path=data_path,
    )
    
    result = engine.analyze(transaction)
    
    if display_result:
        result.display()
    
    return result


def explain_decision(
    transaction: dict[str, Any],
    effects_report_path: str = "causal_effects_report.json",
    language: str = "pl",
) -> str:
    """
    Generate human-readable explanation for a fraud decision.
    
    Args:
        transaction: Transaction features dictionary.
        effects_report_path: Path to causal effects report.
        language: Output language ("pl" or "en").
    
    Returns:
        Human-readable explanation string.
    """
    engine = CounterfactualEngine.from_files(effects_report_path=effects_report_path)
    result = engine.analyze(transaction)
    
    prob_pct = result.baseline_probability * 100
    
    if language == "pl":
        lines = [
            f"📊 ANALIZA TRANSAKCJI {result.transaction_id}",
            f"",
            f"Prawdopodobieństwo fraudu: {prob_pct:.1f}%",
            f"",
            f"Główne czynniki ryzyka:",
        ]
        
        for f in result.get_top_contributors(3):
            contrib_pp = f.contribution * 100
            if contrib_pp > 0.5:
                lines.append(f"  • {f.name} = {f.current_value:.2f} (dodaje {contrib_pp:.1f}pp do ryzyka)")
        
        if result.recommendations:
            lines.append("")
            lines.append("Rekomendacje:")
            for r in result.get_actionable_recommendations()[:2]:
                reduction_pp = r.probability_reduction * 100
                lines.append(f"  → {r.factor_name}: zmień na {r.recommended_value:.2f} (−{reduction_pp:.1f}pp)")
    
    else:  # English
        lines = [
            f"📊 TRANSACTION ANALYSIS {result.transaction_id}",
            f"",
            f"Fraud probability: {prob_pct:.1f}%",
            f"",
            f"Main risk factors:",
        ]
        
        for f in result.get_top_contributors(3):
            contrib_pp = f.contribution * 100
            if contrib_pp > 0.5:
                lines.append(f"  • {f.name} = {f.current_value:.2f} (adds {contrib_pp:.1f}pp to risk)")
        
        if result.recommendations:
            lines.append("")
            lines.append("Recommendations:")
            for r in result.get_actionable_recommendations()[:2]:
                reduction_pp = r.probability_reduction * 100
                lines.append(f"  → {r.factor_name}: change to {r.recommended_value:.2f} (−{reduction_pp:.1f}pp)")
    
    return "\n".join(lines)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("Counterfactual Engine")
    print("=" * 40)
    print()
    print("Usage in Python:")
    print("  from counterfactual_engine import analyze_transaction")
    print()
    print("  result = analyze_transaction({")
    print('      "transaction_amount": 25000,')
    print('      "merchant_risk_score": 0.75,')
    print('      "transaction_velocity_24h": 5,')
    print("  })")
    print()
    print("Or step by step:")
    print("  from counterfactual_engine import CounterfactualEngine")
    print("  engine = CounterfactualEngine.from_files()")
    print("  result = engine.analyze(transaction)")
    print("  result.display()")
