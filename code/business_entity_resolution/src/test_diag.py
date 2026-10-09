"""Label-free diagnostics of the test predictions by country.

Re-scores the cached test chunks with the saved models, then reports per
country: distribution of the top candidate probability per S1, share of S1
whose top probability is uncertain (0.3-0.8), blocked/candidate counts and the
predicted-empty share; prints French examples side by side with candidates.

    python code/business_entity_resolution/src/test_diag.py --artifacts-dir artifacts
"""
import argparse
import glob
import json
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import lightgbm as lgb  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep  # noqa: E402
from pipeline import load_chunk, stage2_frame  # noqa: E402


def work_dirs(artifacts, args):
    cands = [d for d in glob.glob(os.path.join(artifacts, "work", "*"))
             if f"_st{args['store_per_source']}_" in d
             and (d.endswith(f"_ad{args['store_addr']}") if args["store_addr"] else "_ad" not in d)]
    work = max(cands, key=os.path.getmtime)
    return work, max(glob.glob(os.path.join(work, f"f_{args['feature_set']}_*")), key=os.path.getmtime)


def score_test(artifacts, out_path):
    rep = json.load(open(os.path.join(artifacts, "run_report.json")))
    args, t1 = rep["args"], rep["stage1"]["threshold"]
    _, fdir = work_dirs(artifacts, args)
    st1 = lgb.Booster(model_file=os.path.join(artifacts, "model_stage1.txt"))
    if args.get("learner") == "xgb":
        import xgboost as xgb
        from pipeline import XGBModel
        bst = xgb.Booster()
        bst.load_model(os.path.join(artifacts, "model_stage2_xgb.json"))
        st2 = XGBModel(bst, bst.feature_names)
        s2_cols = bst.feature_names
    else:
        st2 = lgb.Booster(model_file=os.path.join(artifacts, "model_stage2.txt"))
        s2_cols = st2.feature_name()
    parts, blocked = [], []
    for f in sorted([x for x in glob.glob(os.path.join(fdir, "test_*.parquet")) if not x.endswith("_emb.parquet")]):
        df = load_chunk(f, args["keep_per_source"], with_c1=True, kpa=args["keep_addr"])
        blocked.append(df.groupby("s1_id").size().rename("n_blocked"))
        p1 = st1.predict(df[st1.feature_name()])
        top_cheap = df.assign(p1=p1).groupby("s1_id").agg(best_cos_name=("cos_name", "max"),
                                                           best_cos_addr=("cos_addr", "max"))
        blocked[-1] = pd.concat([blocked[-1], top_cheap], axis=1)
        k = p1 >= t1
        df = df[k].reset_index(drop=True)
        X = stage2_frame(df, p1[k], args["key_feature"])[s2_cols]
        parts.append(pd.DataFrame({"s1_id": df.s1_id.values, "entity_id": df.entity_id.values,
                                   "bucket": df.bucket.values, "p": st2.predict(X), "p1": p1[k],
                                   "cos_name": df.cos_name.values, "cos_addr": df.cos_addr.values}))
        print("scored", os.path.basename(f), len(df), flush=True)
    P = pd.concat(parts, ignore_index=True)
    B = pd.concat(blocked).groupby(level=0).agg(n_blocked=("n_blocked", "sum"),
                                                best_cos_name=("best_cos_name", "max"),
                                                best_cos_addr=("best_cos_addr", "max"))
    P.to_parquet(out_path, index=False)
    B.to_parquet(out_path.replace(".parquet", "_blocked.parquet"))
    return P, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--out", default="output")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    path = os.path.join(a.artifacts_dir, f"test_pairs{a.tag}.parquet")
    if a.rescore or not os.path.exists(path):
        P, B = score_test(a.artifacts_dir, path)
    else:
        P, B = pd.read_parquet(path), pd.read_parquet(path.replace(".parquet", "_blocked.parquet"))
    t1 = prep.read_tsv(os.path.join(a.data_dir, "test", "test_source1.tsv"))
    m = prep.read_tsv(os.path.join(a.out, "matching_results.tsv")).fillna("")
    s1 = t1[["entity_id", "country"]].rename(columns={"entity_id": "s1_id"})
    top = P.groupby("s1_id")["p"].max().rename("top_p")
    s1 = s1.merge(top, left_on="s1_id", right_index=True, how="left").merge(B, left_on="s1_id",
                                                                            right_index=True, how="left")
    s1["top_p"] = s1["top_p"].fillna(0)
    s1["n_cands"] = s1.s1_id.map(P.groupby("s1_id").size()).fillna(0)
    s1["n_blocked"] = s1["n_blocked"].fillna(0)
    s1["pred_empty"] = s1.s1_id.map(dict(zip(m.source1_entity_id, m.matched_entity_ids == ""))).astype(bool)
    print("\n=== per-country diagnostics (test)")
    g = s1.groupby("country")
    out = pd.DataFrame({
        "S1": g.size(),
        "top_p q10": g.top_p.quantile(0.10), "q25": g.top_p.quantile(0.25), "median": g.top_p.median(),
        "uncertain(0.3-0.8)": g.top_p.apply(lambda x: ((x >= 0.3) & (x < 0.8)).mean()),
        "top_p<0.3": g.top_p.apply(lambda x: (x < 0.3).mean()),
        "avg blocked": g.n_blocked.mean(), "avg cands": g.n_cands.mean(),
        "pred empty": g.pred_empty.mean()})
    print(out.round(4).to_string())

    fr = s1[s1.country == "France"]
    unc = fr[(fr.top_p >= 0.3) & (fr.top_p < 0.8)]
    far = fr[(fr.best_cos_name.fillna(0) < 0.5) & (fr.best_cos_addr.fillna(0) < 0.5)]
    print(f"\nFrance: uncertain-zone S1 {len(unc)}; S1 whose best blocked name AND address cosine < 0.5: {len(far)}")
    ex_u = unc.sample(min(40, len(unc)), random_state=0)
    ex_f = far.sample(min(20, len(far)), random_state=0)
    need_c = P[P.s1_id.isin(set(ex_u.s1_id) | set(ex_f.s1_id))]
    ids = set(need_c.entity_id)
    raws = [t1.set_index("entity_id")[["business_name", "business_address"]]]
    for i in (2, 3):
        for ch in pd.read_csv(os.path.join(a.data_dir, "test", f"test_source{i}.tsv"), sep="\t", dtype=str,
                              keep_default_na=False, quoting=3, chunksize=1000000):
            raws.append(ch[ch.entity_id.isin(ids)].set_index("entity_id")[["business_name", "business_address"]])
    raw = pd.concat(raws)

    def show(df, title):
        print(f"\n===== {title}")
        for r in df.itertuples():
            print(f"S1 top_p={r.top_p:.2f} | {raw.business_name.get(r.s1_id)!s:42.42} | {raw.business_address.get(r.s1_id)!s:.70}")
            c = need_c[need_c.s1_id == r.s1_id].sort_values("p", ascending=False).head(4)
            for q in c.itertuples():
                print(f"   {q.entity_id[:2]} p={q.p:.2f} | {raw.business_name.get(q.entity_id)!s:42.42} | "
                      f"{raw.business_address.get(q.entity_id)!s:.70}")
            if not len(c):
                print("   (no candidate after the stage-1 filter)")
    show(ex_u, "40 French S1 with top probability in 0.3-0.8")
    show(ex_f, "20 French S1 where blocking found nothing close (best name & address cosine < 0.5)")


if __name__ == "__main__":
    main()
