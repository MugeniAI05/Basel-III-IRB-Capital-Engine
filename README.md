# Basel III IRB Capital Engine

A production-grade implementation of the Basel III Internal Ratings-Based (IRB) capital framework, covering PD/LGD modeling, portfolio construction, Vasicek single-factor stress testing, and regulatory capital calculation.

---

## Features

- **Synthetic Portfolio Generation** — 10,000 heterogeneous corporate loans across 8 industries
- **Expected Loss (EL)** — PD × LGD × EAD aggregated to portfolio level
- **Vasicek Single-Factor Model** — Conditional default probabilities under systematic stress
- **IRB Capital Formula** — Full Basel III AIRB implementation with Basel correlation function
- **Sensitivity Analysis** — Capital vs PD, LGD, correlation, confidence level
- **Stress Scenarios** — CCAR-style shocks with capital adequacy assessment
- **Interactive Dashboard** — Streamlit web app with live charts
- **CLI** — Run full analysis from the command line

---

## Project Structure

```
basel3_irb_engine/
├── src/
│   ├── models/
│   │   ├── vasicek.py          # Vasicek single-factor model
│   │   └── irb_capital.py      # Basel III IRB capital formula
│   ├── portfolio/
│   │   ├── generator.py        # Synthetic loan portfolio generation
│   │   └── analytics.py        # Portfolio-level EL, capital, RWA
│   ├── stress/
│   │   └── scenarios.py        # Stress testing & CCAR scenarios
│   ├── visualization/
│   │   └── charts.py           # Reusable Plotly chart builders
│   └── utils/
│       └── math_utils.py       # Normal CDF/InvCDF, statistical helpers
├── tests/
│   ├── test_irb_capital.py
│   ├── test_vasicek.py
│   └── test_portfolio.py
├── notebooks/
│   └── irb_analysis.ipynb      # Jupyter walkthrough
├── app.py                      # Streamlit dashboard entry point
├── main.py                     # CLI entry point
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/basel3-irb-engine.git
cd basel3-irb-engine

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

### 3. Run CLI Analysis

```bash
python main.py
```

Prints full portfolio summary, EL/capital breakdown, and stress scenario results to the terminal.

### 4. Run Tests

```bash
pytest tests/ -v
```

---

## Math Overview

### IRB Capital Formula (Basel III AIRB)

```
K = LGD × Φ[(Φ⁻¹(PD) + √R · Φ⁻¹(0.999)) / √(1-R)] − PD × LGD

RWA     = 12.5 × EAD × K
Capital = 8% × RWA
```

Where `R` is Basel asset correlation:

```
R = 0.12 × (1 - e^(-50·PD)) / (1 - e^(-50))
  + 0.24 × [1 - (1 - e^(-50·PD)) / (1 - e^(-50))]
```

### Vasicek Conditional PD

```
PD(Y) = Φ[(Φ⁻¹(PD) − √R · Y) / √(1-R)]
```

Where `Y ~ N(0,1)` is the systematic factor. `Y = -2` approximates a severe recession.

---

## Model Risk Considerations

- Gaussian copula underestimates tail dependence in crises
- Basel correlations (12–24%) underestimate realized crisis correlations (50–80%)
- TTC vs PIT PD tension introduces procyclicality
- Single-factor model cannot capture sector-specific contagion
- LGD estimates carry substantial uncertainty under stress

---

## License

MIT
