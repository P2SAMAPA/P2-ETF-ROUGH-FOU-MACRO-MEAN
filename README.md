# Rough Fractional Ornstein‑Uhlenbeck (fOU) Process with Macro Mean

Models ETF returns as a rough fractional OU process where the long‑term mean depends on macro variables (VIX, DXY, yields). The per‑ETF score is the current macro‑conditional mean – a signal of expected drift.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Hurst exponent estimation via DFA
- Mean‑reversion speed θ and macro sensitivity β estimated via joint regression
- Score = μ(macro_today) = (intercept + β·macro)/θ
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-rough-fou-macro-mean-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py`
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High score → ETF is expected to drift upward given current macro conditions.
- Low score → expected downward drift.

## Requirements

See `requirements.txt`.
