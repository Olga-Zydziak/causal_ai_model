# 🗺️ ROADMAP: Causal AI + Neurosymbolic System dla Bankowości

## 📍 Status projektu: POC (Proof of Concept) dla AML/Fraud Detection

---

## ✅ ZROBIONE (Etapy 1-3)

### Etap 1: Causal Discovery ✅
**Cel:** Automatyczne odkrywanie struktury przyczynowej z danych

| Komponent | Status | Opis |
|-----------|--------|------|
| `causal_discovery_engine.py` | ✅ | Silnik discovery z 3 algorytmami |
| Algorytm PC | ✅ | Constraint-based (testy niezależności) |
| Algorytm LiNGAM | ✅ | Funkcyjny (non-Gaussian) |
| Algorytm GES | ✅ | Score-based (BIC) |
| Ensemble voting | ✅ | Łączy wyniki 3 algorytmów |
| `discovery_result.json` | ✅ | Output: odkryty graf |

**Wynik:** Graf przyczynowy pokazujący które zmienne wpływają na `is_fraud`

---

### Etap 2: Human-in-the-Loop Review ✅
**Cel:** Ekspert weryfikuje/poprawia odkryty graf

| Komponent | Status | Opis |
|-----------|--------|------|
| `causal_graph_review_svg.py` | ✅ | Interfejs SVG w Jupyter |
| Wizualizacja grafu | ✅ | Kolorowe węzły/krawędzie |
| Approve/Reject | ✅ | Zatwierdzanie krawędzi |
| Mark as confounded | ✅ | Oznaczanie confounders |
| Auto-approve | ✅ | Szybkie zatwierdzenie z Ground Truth |
| Export JSON | ✅ | `approved_graph.json` |

**Wynik:** Zwalidowany graf przyczynowy z eksperckimi poprawkami

---

### Etap 3: Causal Effect Estimation (ATE/CATE) ✅
**Cel:** Obliczenie siły efektów przyczynowych

| Komponent | Status | Opis |
|-----------|--------|------|
| `causal_effect_estimator.py` | ✅ | Silnik estymacji |
| ATE (Average Treatment Effect) | ✅ | Średni efekt każdego czynnika |
| CATE (Conditional ATE) | ✅ | Efekty dla podgrup (segmentów) |
| HTML Report | ✅ | Wizualizacja z heatmapą CATE |
| JSON Export | ✅ | `causal_effects_report.json` |
| Walidacja vs Ground Truth | ✅ | 100% zgodność |

**Wynik:** Ranking czynników ryzyka z dokładnymi wartościami wpływu

---

### Etap 0 (osobny projekt): AxiomKernel ✅
**Cel:** Deterministyczny silnik reguł oparty na Z3 SMT

| Komponent | Status | Opis |
|-----------|--------|------|
| `axiomatic_kernel.py` | ✅ | Core SMT decision engine |
| `nl_rule_parser.py` | ✅ | Parser reguł NL → Z3 |
| `explanation_engine.py` | ✅ | Human-readable wyjaśnienia |
| `rules_io.py` | ✅ | Loader YAML/JSON |
| `ruleset_manager.py` | ✅ | Lifecycle (DEV/PROD) |
| `rule_analytics.py` | ✅ | Statystyki użycia reguł |
| UNSAT Detection | ✅ | Wykrywanie sprzecznych reguł |
| Audit Trail (JSONL) | ✅ | Pełna ścieżka audytowa |

**Wynik:** Production-ready silnik reguł z formalną weryfikacją

---

## 🔄 W TRAKCIE / NASTĘPNE

### Etap 4: Counterfactual Reasoning 🔜
**Cel:** "Co by było gdyby?" dla pojedynczych transakcji

| Komponent | Status | Opis |
|-----------|--------|------|
| `counterfactual_engine.py` | 📋 TODO | Silnik counterfactuals |
| Single transaction analysis | 📋 TODO | Analiza jednej tx |
| "What-if" scenarios | 📋 TODO | Symulacja zmian |
| Minimal intervention | 📋 TODO | "Co zmienić żeby przeszło?" |
| Explainability output | 📋 TODO | Wyjaśnienia dla klienta/regulatora |

**Przykład:**
```
Transakcja #12345 zablokowana (P(fraud)=73%)

Counterfactual:
  → Gdyby velocity było 2 zamiast 7: P(fraud)=31%
  → Gdyby merchant_risk było 0.3 zamiast 0.8: P(fraud)=45%

Rekomendacja: "Zmniejsz liczbę transakcji w 24h"
```

---

### Etap 5: Intervention Simulation 🔜
**Cel:** Symulacja efektu zmian polityk na całej populacji

| Komponent | Status | Opis |
|-----------|--------|------|
| `intervention_simulator.py` | 📋 TODO | Silnik interwencji |
| do(X=value) operator | 📋 TODO | Wymuszenie wartości |
| Policy simulation | 📋 TODO | "Co jeśli zmienimy politykę?" |
| ROI analysis | 📋 TODO | Koszt vs benefit |
| What-if dashboards | 📋 TODO | Interaktywne scenariusze |

**Przykład:**
```
Scenariusz: do(block_foreign=TRUE) - blokujemy zagraniczne tx

Wynik:
  - Fraudy: -18%
  - Utracone przychody: -5%
  - ROI: +340%
```

---

### Etap 6: Integracja Causal ↔ AxiomKernel 🔜
**Cel:** Połączenie obu systemów w hybrid engine

| Komponent | Status | Opis |
|-----------|--------|------|
| `causal_rule_generator.py` | 📋 TODO | Auto-generowanie reguł YAML z grafu |
| `causal_explainer.py` | 📋 TODO | Counterfactuals w Explanation Engine |
| `rule_validator.py` | 📋 TODO | Walidacja reguł przez kauzalność |
| `hybrid_decision_engine.py` | 📋 TODO | Unified pipeline |

**Przepływ:**
```
Dane → Causal Discovery → ATE/CATE → Auto-generated Rules → AxiomKernel
                                              ↓
                                    Formal Verification (Z3)
                                              ↓
                                    Decision + Explanation
```

---

## 📋 BACKLOG (Przyszłość)

### Etap 7: Regulatory Knowledge Compiler
**Cel:** Ekstrakcja reguł z dokumentów prawnych (LLM + Z3)

| Komponent | Opis |
|-----------|------|
| PDF/DOCX parser | Ekstrakcja tekstu z regulacji |
| LLM rule extraction | NL → structured rules |
| Provenance tracking | Która reguła z którego dokumentu |
| Conflict detection | Sprzeczności między regulacjami |

---

### Etap 8: Production Deployment
**Cel:** Wdrożenie na środowisko produkcyjne

| Komponent | Opis |
|-----------|------|
| REST API | FastAPI wrapper |
| Monitoring | Prometheus/Grafana |
| A/B testing | Porównanie z baseline |
| Model versioning | MLflow/DVC |

---

### Etap 9: Continuous Learning
**Cel:** Automatyczna aktualizacja modelu

| Komponent | Opis |
|-----------|------|
| Drift detection | Wykrywanie zmian w danych |
| Auto-retraining | Periodic re-discovery |
| Feedback loop | Human corrections → retrain |

---

## 🎯 PRIORYTETY NA TERAZ

### Opcja A: Counterfactuals (Etap 4)
**Dla kogo:** Jeśli chcesz wyjaśniać pojedyncze decyzje
```
"Dlaczego zablokowano TĘ transakcję?"
"Co klient musi zmienić żeby przeszło?"
```

### Opcja B: Integracja z AxiomKernel (Etap 6)
**Dla kogo:** Jeśli chcesz połączyć oba systemy w jeden POC
```
Causal AI → Auto-generated Rules → Z3 Verification → Decision
```

### Opcja C: Intervention Simulation (Etap 5)
**Dla kogo:** Jeśli chcesz pokazać wartość biznesową
```
"Ile fraudów unikniemy jeśli zmienimy politykę X?"
"Jaki jest ROI tej zmiany?"
```

---

## 📊 ARCHITEKTURA DOCELOWA

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     HYBRID CAUSAL-SYMBOLIC SYSTEM                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │   DATA      │    │   CAUSAL    │    │  EFFECTS    │                 │
│  │  (50k tx)   │ →  │  DISCOVERY  │ →  │  (ATE/CATE) │                 │
│  └─────────────┘    └─────────────┘    └──────┬──────┘                 │
│                                               │                         │
│                     ┌─────────────────────────┼─────────────────────┐  │
│                     │                         ▼                     │  │
│                     │    ┌─────────────────────────────────────┐   │  │
│                     │    │      CAUSAL RULE GENERATOR          │   │  │
│                     │    │   (ATE → YAML rules automatically)  │   │  │
│                     │    └─────────────────┬───────────────────┘   │  │
│                     │                      │                        │  │
│                     │                      ▼                        │  │
│  ┌─────────────┐    │    ┌─────────────────────────────────────┐   │  │
│  │  HUMAN      │ ←──┼──→ │         AXIOM KERNEL (Z3)           │   │  │
│  │  REVIEW     │    │    │  • Formal verification              │   │  │
│  └─────────────┘    │    │  • Conflict detection               │   │  │
│                     │    │  • Proof-carrying decisions         │   │  │
│                     │    └─────────────────┬───────────────────┘   │  │
│                     │                      │                        │  │
│                     │                      ▼                        │  │
│                     │    ┌─────────────────────────────────────┐   │  │
│                     │    │       EXPLANATION ENGINE             │   │  │
│                     │    │  • Why was tx blocked?              │   │  │
│                     │    │  • Counterfactual reasoning         │   │  │
│                     │    │  • Regulatory compliance            │   │  │
│                     │    └─────────────────────────────────────┘   │  │
│                     │                                               │  │
│                     │              INTEGRATION LAYER                │  │
│                     └───────────────────────────────────────────────┘  │
│                                                                         │
│  OUTPUT:                                                                │
│    • Deterministic decisions (SAT/UNSAT)                               │
│    • Human-readable explanations                                        │
│    • Audit trail (JSONL)                                               │
│    • Counterfactual recommendations                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ❓ CO DALEJ?

Który etap chcesz zrobić następny?

1. **Etap 4: Counterfactuals** - wyjaśnienia "co by było gdyby"
2. **Etap 5: Intervention Simulation** - symulacja polityk
3. **Etap 6: Integracja** - połączenie z AxiomKernel

---

*Ostatnia aktualizacja: 2025-12-10*
