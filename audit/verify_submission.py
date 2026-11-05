"""Validate a prepared release without training or changing submitted predictions."""
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    sys.path.insert(0, str(ROOT))
    import score

    train = pd.read_csv(ROOT / "data/train_test.csv", parse_dates=["date"])
    final = pd.read_csv(ROOT / "data/validation.csv", parse_dates=["date"])
    predictions = pd.read_csv(ROOT / "validation_predictions.csv")
    december = pd.read_csv(ROOT / "data/december_chart_inputs.csv")
    template = pd.read_csv(ROOT / "data/validation_predictions_template.csv")
    assert len(train) == 48000 and len(final) == 12000
    assert train.date.min() == pd.Timestamp("2025-01-01") and train.date.max() == pd.Timestamp("2025-10-31")
    assert final.date.min() == pd.Timestamp("2025-11-01") and final.date.max() == pd.Timestamp("2025-12-31")
    assert train.load_id.is_unique and final.load_id.is_unique
    assert not (set(train.load_id) & set(final.load_id))
    assert "posted_rate" in train and "posted_rate" not in final
    assert np.isfinite(train.posted_rate).all() and train.posted_rate.gt(0).all()
    feature_columns = {"pickup", "delivery", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
                       "equipment", "distance", "weight", "date", "market_index", "quote_signal", "load_id"}
    assert feature_columns.issubset(train.columns) and feature_columns.issubset(final.columns)
    score.validate_predictions(predictions)
    score.validate_december(december)
    assert set(predictions.load_id) == set(final.load_id) == set(template.load_id)
    assert predictions.load_id.tolist() == template.load_id.tolist()
    syntax_files = sorted(list((ROOT / "src").glob("*.py")) + list((ROOT / "audit").glob("*.py")) + [ROOT / "score.py"])
    for path in syntax_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path.relative_to(ROOT)))
    required = ["README.md", "requirements.txt", "requirements-lock.txt", "scorer_results/candidate_december.png",
                "output/pdf/Spotter_Freight_Rate_Assessment_Report.pdf", "docs/assessment_instructions.pdf"]
    for name in required:
        assert (ROOT / name).is_file(), f"Missing release file: {name}"
    assert (ROOT / required[-2]).read_bytes().startswith(b"%PDF")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)]+)\)", readme):
        if not target.startswith(("https://", "http://", "#")):
            assert (ROOT / target.split("#")[0]).is_file(), f"Broken README link: {target}"
    manifest_path = ROOT / "submission_manifest.json"
    assert manifest_path.is_file(), "The prepared release needs submission_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT), f"Manifest path outside release: {name}"
        assert path.is_file(), f"Missing manifest file: {name}"
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, f"Release checksum differs: {name}"
    summary = {"status": "PASS", "training_performed": False, "prediction_files_modified": False,
               "labeled_rows": len(train), "final_rows": len(final), "prediction_rows": len(predictions),
               "december_rows": len(december), "manifest_files_verified": len(manifest["sha256"]),
               "python_files_parsed": len(syntax_files), "readme_links_valid": True,
               "original_scorer_validations": "PASS", "prediction_order_matches_template": True}
    output = ROOT / "audit/submission_checks"
    output.mkdir(parents=True, exist_ok=True)
    (output / "verification.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
