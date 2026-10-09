"""Compare the threshold decision rule with the expected-F0.5 rule on OOF.

    python code/business_entity_resolution/src/decision_eval.py --artifacts-dir artifacts
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decision import expected_f05_keep  # noqa: E402
from pipeline import decide_mask, in_sample, macro_f05, resolve_conflicts  # noqa: E402
from prep import read_tsv  # noqa: E402


def nested_isotonic(p, y, groups, folds=5):
    """Calibrate OOF probabilities: each fold's calibrator is fitted on the other folds."""
    out = np.empty_like(p)
    for a, b in GroupKFold(n_splits=folds).split(p, y, groups):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[a], y[a])
        out[b] = iso.predict(p[b])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--artifacts-dir", default="artifacts")
    a = ap.parse_args()
    rep = json.load(open(os.path.join(a.artifacts_dir, "run_report.json")))
    thr, alpha = rep["validation"]["threshold"], rep["validation"]["alpha"]
    oo = pd.read_parquet(os.path.join(a.artifacts_dir, "oof_pairs.parquet"))
    gt = read_tsv(os.path.join(a.data_dir, "train", "train_ground_truth.tsv")).fillna("")
    s1 = gt.source1_entity_id.values
    samp = in_sample(s1, rep["args"]["train_frac"])
    ids = s1[samp]
    ntrue = np.array([len([x for x in v.split(",") if x]) for v in gt.matched_entity_ids.values[samp]], float)
    code = pd.Series(np.arange(len(ids)), index=ids).reindex(oo.s1_id.values).values
    y, p = oo.y.values.astype(int), oo.p.values
    sim = (oo.cos_name + oo.cos_addr).values
    eid = oo.entity_id.values

    print("reliability (OOF probability bins):")
    bins = np.clip((p * 10).astype(int), 0, 9)
    t = pd.DataFrame({"bin": bins, "p": p, "y": y}).groupby("bin").agg(n=("y", "size"), mean_p=("p", "mean"),
                                                                      frac_true=("y", "mean"))
    print(t.round(4).to_string())
    print(f"overall: mean p {p.mean():.4f}, frac true {y.mean():.4f}")

    def report(name, keep):
        f = macro_f05(code, y, keep, ntrue)
        kr, info = resolve_conflicts(code, eid, p, sim, keep)
        fr = macro_f05(code, y, kr, ntrue)
        print(f"{name:55s} F0.5={f:.5f}  +conflicts={fr:.5f}  pred pairs={int(keep.sum())}", flush=True)
        return f

    base = report(f"threshold rule (t={thr:.3f}, alpha={alpha})", decide_mask(code, p, thr, alpha))
    t0 = time.time()
    pc = nested_isotonic(p, y, code)
    print(f"(nested isotonic calibration in {time.time() - t0:.0f}s; mean p {pc.mean():.4f})")
    tc = pd.DataFrame({"bin": np.clip((pc * 10).astype(int), 0, 9), "p": pc, "y": y}).groupby("bin").agg(
        n=("y", "size"), mean_p=("p", "mean"), frac_true=("y", "mean"))
    print(tc.round(4).to_string())
    res = {}
    for name, prob in (("raw", p), ("isotonic", pc)):
        for miss in (0.0, 0.05, 0.15):
            t0 = time.time()
            keep = expected_f05_keep(code, prob, miss)
            res[(name, miss)] = report(f"expected-F0.5 ({name} p, missed-match prior {miss})", keep)
    best = max(res, key=res.get)
    print(f"\nbest expected-F rule: {best} F0.5={res[best]:.5f} vs threshold {base:.5f} "
          f"(delta {res[best] - base:+.5f})")


if __name__ == "__main__":
    main()
