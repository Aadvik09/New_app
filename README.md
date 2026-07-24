# BootstrapMD

BootstrapMD is a Streamlit workspace for comparing practical synthetic-data methods on CSV research datasets.

## Included workflow

1. Upload a CSV and review inferred types.
2. Confirm included columns and data-type handling.
3. Review validity/constraint warnings.
4. Generate outputs with smoothed bootstrap, SMOTE-NC, Gaussian copula, and CART sequential synthesis.
5. Compare fidelity, predictive utility, and empirical privacy signals; download every synthetic dataset and scorecard.

## Run locally

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

## Important limits

- The privacy panel reports empirical nearest-neighbor and membership-inference-style signals. It is **not** a differential-privacy guarantee.
- Never upload PHI or other sensitive records to an unapproved environment.
- Scores are comparative diagnostics, not clinical validation. Small datasets can yield unstable estimates.
- The included SMOTE-NC routine is a dependency-light mixed-type implementation intended for research exploration; validate outputs for the target use case.
