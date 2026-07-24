import numpy as np
import pandas as pd

from synth_engine import (
    cart_sequential,
    evaluate,
    gaussian_copula,
    infer_schema,
    recommend,
    schema_types,
    smote_nc,
    smoothed_bootstrap,
    validate_data,
)


def fixture_data():
    rng = np.random.default_rng(8)
    return pd.DataFrame({"age": rng.integers(18, 90, 90), "bmi": rng.normal(26, 5, 90), "site": rng.choice(["North", "South", "West"], 90), "outcome": rng.choice(["No", "Yes"], 90, p=[.68, .32])})


def test_full_pipeline_for_all_methods():
    source = fixture_data()
    schema = infer_schema(source)
    types = schema_types(schema)
    assert not [issue for issue in validate_data(source, types) if issue["severity"] == "error"]
    methods = [smoothed_bootstrap(source, types, 120), smote_nc(source, types, 120, "outcome"), gaussian_copula(source, types, 120), cart_sequential(source, types, 120)]
    evaluations = [evaluate(source, output, types, "outcome", f"method {i}") for i, output in enumerate(methods)]
    for output, result in zip(methods, evaluations):
        assert output.shape == (120, 4)
        assert list(output.columns) == list(source.columns)
        assert output["site"].notna().all()
        assert 0 <= result.fidelity <= 100
        assert 0 <= result.privacy <= 100
        assert result.utility is not None
    assert recommend(evaluations, "Balanced").method.startswith("method")
