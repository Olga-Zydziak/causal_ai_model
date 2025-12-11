# 📁 CAUSAL AI FOR FRAUD DETECTION
## Kompletny przewodnik po projekcie

---

## 🗂️ STRUKTURA PLIKÓW

```
causal_ai_project/
│
├── 📊 DANE
│   ├── synthetic_data/
│   │   ├── fraud_synthetic_data.csv      ← 50k transakcji (dane wejściowe)
│   │   ├── ground_truth_metadata.json    ← prawdziwe relacje (do walidacji)
│   │   ├── adjacency_matrix.npz          ← macierz sąsiedztwa
│   │   ├── load_data.py                  ← helper do ładowania
│   │   └── README.md
│   │
│   └── fraud_data.csv                    ← kopia danych w głównym katalogu (opcjonalna)
│
├── 🔧 MODUŁY PYTHONA (4 pliki)
│   ├── causal_discovery_engine.py        ← Etap 1: odkrywanie grafu
│   ├── causal_graph_review_svg.py        ← Etap 2: review UI
│   ├── causal_effect_estimator.py        ← Etap 3: obliczanie ATE/CATE
│   └── counterfactual_engine.py          ← Etap 4: "co by było gdyby"
│
├── 📓 NOTEBOOKI (2 pliki)
│   ├── quick_start.ipynb                 ← cały pipeline w jednym miejscu
│   └── validation.ipynb                  ← walidacja discovery vs ground truth
│
├── 📄 PLIKI WYJŚCIOWE (generowane)
│   ├── discovery_result.json             ← wynik discovery (graf)
│   ├── approved_graph.json               ← zatwierdzony graf (po review)
│   ├── causal_effects_report.json        ← wyniki ATE/CATE
│   └── counterfactual_analysis.json      ← wyniki counterfactual (opcjonalny)
│
└── 📝 DOKUMENTACJA
    └── ROADMAP.md                        ← plan projektu
```

---

## 🔄 PIPELINE - KROK PO KROKU

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   ETAP 1                ETAP 2              ETAP 3              ETAP 4     │
│   Discovery      →      Review       →      Effects      →      Counterfactual
│                                                                             │
│   ┌─────────┐         ┌─────────┐         ┌─────────┐         ┌─────────┐  │
│   │ Dane    │         │ Graf    │         │ Approved│         │ Effects │  │
│   │ CSV     │ ──────► │ JSON    │ ──────► │ Graf    │ ──────► │ Report  │  │
│   └─────────┘         └─────────┘         └─────────┘         └─────────┘  │
│                                                                             │
│   INPUT:              OUTPUT:             OUTPUT:             OUTPUT:       │
│   fraud_data.csv      discovery_          approved_           Wizualizacja  │
│                       result.json         graph.json          + rekomendacje│
│                                           +                                 │
│                                           causal_effects_                   │
│                                           report.json                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔍 WALIDACJA (opcjonalnie)

Po uruchomieniu Discovery możesz sprawdzić jakość wyników:

### Wymagane pliki
```
✓ validation.ipynb
✓ discovery_result.json (z Etapu 1)
✓ synthetic_data/ground_truth_metadata.json
```

### Co pokazuje
```
   Precision: 87.5%  ← ile z odkrytych krawędzi jest prawdziwych
   Recall:    100%   ← ile z prawdziwych krawędzi odkryto
   F1 Score:  93.3%  ← ogólna jakość
   
   ✅ True Positives:  poprawnie odkryte relacje
   ❌ False Positives: fałszywe relacje (spurious)
   ⚠️ False Negatives: pominięte prawdziwe relacje
```

### Ground Truth zawiera:
- 7 obserwowalnych krawędzi przyczynowych
- 2 ukryte krawędzie (confounder: criminal_intent)
- 3 znane fałszywe korelacje (spurious)

---

## 📋 ETAP 1: CAUSAL DISCOVERY

### Cel
Odkrycie struktury przyczynowej z danych (który czynnik wpływa na co?)

### Wymagane pliki
```
✓ causal_discovery_engine.py
✓ synthetic_data/fraud_synthetic_data.csv (lub fraud_data.csv)
```

### Kod w notebooku
```python
import pandas as pd
import json
import numpy as np
from causal_discovery_engine import discover_causal_graph

# Załaduj dane
df = pd.read_csv("fraud_data.csv")

# Opcjonalnie: załaduj ground truth do walidacji
with open("synthetic_data/ground_truth_metadata.json") as f:
    gt = json.load(f)
gt_matrix = np.array(gt["ground_truth"]["adjacency_matrix"])
gt_variables = gt["ground_truth"]["variable_order"]

# Uruchom discovery
result = discover_causal_graph(
    data=df,
    algorithm="lingam",  # lub 'pc', 'ges'
    ground_truth=gt_matrix,
    ground_truth_variables=gt_variables,
    print_report=True,
)

# Zapisz wynik
discovery_output = {
    "algorithm": "lingam",
    "discovered_graph": {
        "edges": [
            {"source": e.source, "target": e.target, "strength": float(e.strength) if e.strength else 1.0}
            for e in result.edges
        ],
        "variables": result.variable_names,
    }
}
with open("discovery_result.json", "w") as f:
    json.dump(discovery_output, f, indent=2)
```

### Output
```
discovery_result.json  ← graf z odkrytymi krawędziami
```

---

## 📋 ETAP 2: HUMAN-IN-THE-LOOP REVIEW

### Cel
Ekspert zatwierdza/odrzuca odkryte relacje przyczynowe

### Wymagane pliki
```
✓ causal_graph_review_svg.py
✓ discovery_result.json (z Etapu 1)
✓ synthetic_data/ground_truth_metadata.json (opcjonalnie, do walidacji)
```

### Kod w notebooku
```python
from causal_graph_review_svg import review_causal_graph

# Uruchom interfejs
reviewer = review_causal_graph(
    discovery_path="discovery_result.json",
    ground_truth_path="synthetic_data/ground_truth_metadata.json",
)

# Po review w UI - zapisz
reviewer.save_approved("approved_graph.json")
```

### Output
```
approved_graph.json  ← zatwierdzony graf
```

---

## 📋 ETAP 3: CAUSAL EFFECT ESTIMATION (ATE/CATE)

### Cel
Obliczenie siły efektów: "O ile X wpływa na fraud?"

### Wymagane pliki
```
✓ causal_effect_estimator.py
✓ approved_graph.json (z Etapu 2)
✓ fraud_data.csv (dane)
```

### Kod w notebooku
```python
from causal_effect_estimator import CausalEffectEstimator

# Utwórz estimator
estimator = CausalEffectEstimator.from_files(
    approved_graph_path="approved_graph.json",
    data_path="fraud_data.csv",
    outcome_variable="is_fraud",
)

# Oblicz efekty
report = estimator.estimate_all()

# Wyświetl raport
report.display()

# Zapisz
report.to_json("causal_effects_report.json")
report.to_html("causal_effects_report.html")
```

### Output
```
causal_effects_report.json  ← wyniki ATE/CATE
causal_effects_report.html  ← raport wizualny (opcjonalny)
```

---

## 📋 ETAP 4: COUNTERFACTUAL REASONING

### Cel
"Co by było gdyby?" dla pojedynczych transakcji

### Wymagane pliki
```
✓ counterfactual_engine.py
✓ causal_effects_report.json (z Etapu 3)
✓ fraud_data.csv (dane - do obliczenia percentyli)
```

### Kod w notebooku
```python
from counterfactual_engine import analyze_transaction, CounterfactualEngine

# Szybka analiza jedną komendą
result = analyze_transaction({
    "transaction_amount": 15000,
    "merchant_risk_score": 0.75,
    "transaction_velocity_24h": 6,
    "account_age_days": 60,
    "is_foreign_transaction": 1,
    "device_fingerprint_age_days": 10,
})

# Lub krok po kroku
engine = CounterfactualEngine.from_files(
    effects_report_path="causal_effects_report.json",
    data_path="fraud_data.csv",
)

# Analiza
result = engine.analyze(transaction, "TX_001")
result.display()

# Counterfactual
new_prob, change = engine.counterfactual(
    transaction,
    {"is_foreign_transaction": 0}
)

# Rekomendacje
recommendations = engine.recommend(transaction, target_probability=0.30)
```

### Output
```
Wizualizacja w Jupyter
counterfactual_analysis.json (opcjonalny)
```

---

## ✅ LISTA KONTROLNA - MINIMALNE WYMAGANIA

### Pliki które MUSISZ mieć:

| Plik | Skąd go wziąć | Do czego służy |
|------|---------------|----------------|
| `causal_discovery_engine.py` | z paczki ZIP | Etap 1 |
| `causal_graph_review_svg.py` | z paczki ZIP | Etap 2 |
| `causal_effect_estimator.py` | z paczki ZIP | Etap 3 |
| `counterfactual_engine.py` | z paczki ZIP | Etap 4 |
| `fraud_data.csv` | `synthetic_data/` lub własne | dane wejściowe |

### Pliki generowane (nie kopiuj - generuj!):

| Plik | Generowany przez | Etap |
|------|------------------|------|
| `discovery_result.json` | `run_causal_discovery()` | 1 |
| `approved_graph.json` | `reviewer.save_approved()` | 2 |
| `causal_effects_report.json` | `report.to_json()` | 3 |

---

## 🚀 SZYBKI START (wszystko w jednym notebooku)

```python
# ============================================================
# ETAP 1: Discovery
# ============================================================
import pandas as pd
import json
import numpy as np
from causal_discovery_engine import discover_causal_graph

df = pd.read_csv("fraud_data.csv")

# Opcjonalnie: ground truth do walidacji
with open("synthetic_data/ground_truth_metadata.json") as f:
    gt = json.load(f)
gt_matrix = np.array(gt["ground_truth"]["adjacency_matrix"])
gt_variables = gt["ground_truth"]["variable_order"]

result = discover_causal_graph(
    data=df,
    algorithm="lingam",
    ground_truth=gt_matrix,
    ground_truth_variables=gt_variables,
)

# Zapisz
discovery_output = {
    "algorithm": "lingam",
    "discovered_graph": {
        "edges": [{"source": e.source, "target": e.target, "strength": float(e.strength or 1.0)} for e in result.edges],
        "variables": result.variable_names,
    }
}
with open("discovery_result.json", "w") as f:
    json.dump(discovery_output, f, indent=2)
print("✓ Etap 1 done")

# ============================================================
# ETAP 2: Review (auto-approve dla demo)
# ============================================================
from causal_graph_review_svg import review_causal_graph, EdgeStatus

reviewer = review_causal_graph(
    discovery_path="discovery_result.json",
    ground_truth_path="synthetic_data/ground_truth_metadata.json",
)

# Auto-approve wszystko
for edge in reviewer.edges.values():
    edge.status = EdgeStatus.APPROVED
    edge.approved_strength = edge.ground_truth or edge.discovered_strength

reviewer.save_approved("approved_graph.json")
print("✓ Etap 2 done")

# ============================================================
# ETAP 3: Effects
# ============================================================
from causal_effect_estimator import CausalEffectEstimator

estimator = CausalEffectEstimator.from_files(
    approved_graph_path="approved_graph.json",
    data_path="fraud_data.csv",
    outcome_variable="is_fraud",
)

report = estimator.estimate_all()
report.to_json("causal_effects_report.json")
report.display()
print("✓ Etap 3 done")

# ============================================================
# ETAP 4: Counterfactual
# ============================================================
from counterfactual_engine import analyze_transaction

result = analyze_transaction({
    "transaction_amount": 15000,
    "merchant_risk_score": 0.75,
    "transaction_velocity_24h": 6,
    "account_age_days": 60,
    "is_foreign_transaction": 1,
    "device_fingerprint_age_days": 10,
})
print("✓ Etap 4 done")
```

---

## ⚠️ TYPOWE PROBLEMY

### Problem: `approved_graph.json` jest pusty
**Rozwiązanie:** Musisz kliknąć "Approve" lub użyć auto-approve PRZED `save_approved()`

### Problem: `FileNotFoundError: causal_effects_report.json`
**Rozwiązanie:** Najpierw uruchom Etap 3 żeby wygenerować ten plik

### Problem: Kernel restart kasuje stan review
**Rozwiązanie:** Zapisz `approved_graph.json` PRZED restartem kernela

### Problem: Brak danych
**Rozwiązanie:** Użyj `fraud_data.csv` z katalogu `synthetic_data/`

---

## 📦 AKTUALNA PACZKA

Pobierz: `causal_ai_v6_datadriven.zip`

Zawiera wszystkie 4 moduły + notebooki demo + dane syntetyczne.

---

*Ostatnia aktualizacja: 2025-12-10*
