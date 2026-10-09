"""Evaluate one-owner conflict resolution on train (no retraining).

Every S2/S3 record belongs to at most one S1 entity (true in the train ground
truth). The matcher scores each S1 independently, so a record can be claimed by
several S1 entities. `resolve_conflicts` keeps each claimed record only for the
S1 entity with the highest probability (ties -> higher name+address cosine).

Evaluation universe: the sampled S1 of the last run (out-of-fold stage-2
probabilities). Their competitors are mostly *non-sampled* S1, which the final
models never saw in training, so those are scored here with the final models
(held-out predictions) and take part in the conflicts.

    python code/business_entity_resolution/src/conflict_eval.py --artifacts-dir artifacts
"""
import argparse
import glob
import json
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep  # noqa: E402
from pipeline import (decide_mask, in_sample, load_chunk, macro_f05, pair_key,  # noqa: E402
                      resolve_conflicts, stage2_frame)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("-n", type=int, default=20, help="examples to print")
    a = ap.parse_args()
    rep = json.load(open(os.path.join(a.artifacts_dir, "run_report.json")))
    args = rep["args"]
    thr, alpha, t1 = rep["validation"]["threshold"], rep["validation"]["alpha"], rep["stage1"]["threshold"]
    cache = os.path.join(a.artifacts_dir, "cache")
    s1_path = prep.ensure_cache(os.path.join(a.data_dir, "train", "train_source1.tsv"), cache)
    s1_ids = prep.col(prep.read_bucket(s1_path, "__missing__"), "entity_id")
    samp = in_sample(s1_ids, args["train_frac"])
    gt = prep.read_tsv(os.path.join(a.data_dir, "train", "train_ground_truth.tsv")).fillna("")
    ex = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    ex = ex[ex.m != ""]
    true_keys = np.sort(pair_key(ex.source1_entity_id.values, ex.m.values))
    ntrue_map = dict(zip(gt.source1_entity_id, gt.matched_entity_ids.map(lambda v: len([x for x in v.split(",") if x]))))

    # --- out-of-fold pairs of the sampled S1
    oo = pd.read_parquet(os.path.join(a.artifacts_dir, "oof_pairs.parquet"))
    code = pd.Series(np.arange(len(s1_ids)), index=s1_ids)
    parts = [pd.DataFrame({"s1": code.reindex(oo.s1_id.values).values, "eid": oo.entity_id.values,
                           "p": oo.p.values, "y": oo.y.values.astype(int),
                           "sim": (oo.cos_name + oo.cos_addr).values, "oof": True})]

    # --- held-out predictions for the non-sampled S1 (final models)
    tag = [d for d in glob.glob(os.path.join(a.artifacts_dir, "work", "*"))
           if d.endswith(f"_ad{args['store_addr']}") or (not args["store_addr"] and "_ad" not in d)]
    work = [d for d in tag if f"_st{args['store_per_source']}_" in d][0]
    fdir = max(glob.glob(os.path.join(work, f"f_{args['feature_set']}_*")), key=os.path.getmtime)
    st1 = lgb.Booster(model_file=os.path.join(a.artifacts_dir, "model_stage1.txt"))
    st2 = lgb.Booster(model_file=os.path.join(a.artifacts_dir, "model_stage2.txt"))
    s1c, s2c = st1.feature_name(), st2.feature_name()
    for f in sorted([x for x in glob.glob(os.path.join(fdir, "train_*.parquet")) if not x.endswith("_emb.parquet")]):
        df = load_chunk(f, args["keep_per_source"], with_c1=True, kpa=args["keep_addr"])
        df = df[~samp[df.s1_row.values]].reset_index(drop=True)
        p1 = st1.predict(df[s1c])
        df = df[p1 >= t1].reset_index(drop=True)
        X = stage2_frame(df, p1[p1 >= t1], args["key_feature"])[s2c]
        y = np.isin(pair_key(df.s1_id.values, df.entity_id.values), true_keys).astype(int)
        parts.append(pd.DataFrame({"s1": df.s1_row.values, "eid": df.entity_id.values,
                                   "p": st2.predict(X), "y": y,
                                   "sim": (df.cos_name + df.cos_addr).values, "oof": False}))
        print("scored", os.path.basename(f), len(df), flush=True)
        del df, X
    P = pd.concat(parts, ignore_index=True)
    del parts

    # evaluation on the sampled S1 (all of them, incl. those without candidates)
    samp_idx = np.where(samp)[0]
    remap = np.full(len(s1_ids), -1)
    remap[samp_idx] = np.arange(len(samp_idx))
    ntrue = np.array([ntrue_map.get(k, 0) for k in s1_ids[samp_idx]], float)
    ev = P.oof.values

    def score(keep):
        return macro_f05(remap[P.s1.values[ev]], P.y.values[ev], keep[ev], ntrue)

    print(f"\npairs scored: {len(P)} ({ev.sum()} out-of-fold for {len(samp_idx)} sampled S1)")
    for t in sorted({thr, 0.6, 0.65, 0.7, 0.75, 0.8}):
        base = decide_mask(P.s1.values, P.p.values, t, alpha)
        res, info = resolve_conflicts(P.s1.values, P.eid.values, P.p.values, P.sim.values, base)
        print(f"t={t:.3f}: F0.5 without={score(base):.5f}  with={score(res):.5f}  "
              f"(claimed records with >1 S1: {info['conflict_records']}, pairs dropped: "
              f"{info['dropped']}, ties: {info['ties']})")

    base = decide_mask(P.s1.values, P.p.values, thr, alpha)
    res, info = resolve_conflicts(P.s1.values, P.eid.values, P.p.values, P.sim.values, base)
    n_pred = int(base.sum())
    drop = base & ~res
    print(f"\nat the tuned threshold {thr:.3f}: predicted pairs {n_pred}; records claimed by >1 S1: "
          f"{info['conflict_records']}; pairs involved {info['pairs_in_conflict']} "
          f"({info['pairs_in_conflict'] / n_pred:.3%} of predicted matches); dropped {int(drop.sum())} "
          f"of which true matches {int(P.y.values[drop].sum())}; ties {info['ties']}")
    kept_in_conflict = res & P.eid.isin(P.eid.values[drop]).values
    print(f"kept winners in conflicts: {int(kept_in_conflict.sum())}, of which true: "
          f"{int(P.y.values[kept_in_conflict].sum())}")

    # examples
    conf_ids = pd.unique(P.eid.values[drop])
    rs = np.random.RandomState(0)
    ex_ids = set(rs.choice(conf_ids, min(a.n, len(conf_ids)), replace=False))
    sub = P[P.eid.isin(ex_ids) & base].copy()
    sub["kept"] = res[sub.index]
    sub["s1_id"] = s1_ids[sub.s1.values]
    need = set(sub.s1_id) | set(sub.eid)
    raws = []
    for i in (1, 2, 3):
        for ch in pd.read_csv(os.path.join(a.data_dir, "train", f"train_source{i}.tsv"), sep="\t",
                              dtype=str, keep_default_na=False, quoting=3, chunksize=1000000):
            raws.append(ch[ch.entity_id.isin(need)])
    raw = pd.concat(raws).set_index("entity_id")
    print(f"\n===== {len(ex_ids)} resolved conflicts (record claimed by several S1; 'KEEP' = winner)")
    for e, g in sub.groupby("eid"):
        print(f"{e}: {raw.business_name.get(e)!s:45.45} | {raw.business_address.get(e)!s:.70}")
        for r in g.sort_values("p", ascending=False).itertuples():
            print(f"   {'KEEP' if r.kept else 'drop'} p={r.p:.3f} true={r.y} {r.s1_id}: "
                  f"{raw.business_name.get(r.s1_id)!s:40.40} | {raw.business_address.get(r.s1_id)!s:.60}")


if __name__ == "__main__":
    main()
