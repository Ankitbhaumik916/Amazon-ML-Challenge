"""End-to-end pipeline: data -> blocking -> features -> LightGBM -> outputs.

Usage (from student_resource/):
    python code/business_entity_resolution/src/pipeline.py \
        --data-dir dataset --out-dir output --artifacts-dir artifacts

Scales to the full data (millions of records) by working one country bucket x
target source at a time ("chunk"): each chunk is blocked, featurised and
written to parquet under <artifacts>/work, so peak memory is one bucket.
"""
import os

# numeric work here is sparse / LightGBM (OpenMP); a multi-threaded BLAS only
# reserves per-thread buffers in every worker process
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep  # noqa: E402
from blocking import cheap_features, generate_candidates  # noqa: E402
from features import BucketTokens, base_features, context_features, record_context  # noqa: E402

SEED = 42
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63,
                  min_child_samples=20, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=SEED, num_threads=0)
USE_EMB = False
STAGE1_BASE = ["cos_name", "cos_skel", "cos_addr", "cheap", "cheap_rank", "is_s3"]
META = ["s1_row", "i2", "src", "entity_id", "s1_id", "bucket"]


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def id_num(ids):
    """'S2-12345' -> 12345 (ids are unique within a source)."""
    return pd.Series(ids).str.slice(3).astype(np.int64).values


def pair_key(s1_ids, other_ids):
    """Collision-free int64 key for an (S1 id, S2/S3 id) pair."""
    o = pd.Series(other_ids)
    src = np.where(o.str.startswith("S3").values, 2, 1).astype(np.int64)
    return id_num(s1_ids) * np.int64(4_000_000_000) + src * np.int64(1_000_000_000) + id_num(o)


def in_sample(s1_ids, frac):
    """Deterministic S1 sample used for model training / OOF evaluation."""
    if frac >= 1:
        return np.ones(len(s1_ids), bool)
    return (id_num(s1_ids) * 2654435761 % 1_000_003) < frac * 1_000_003


# ------------------------------------------------------------------ metric
def f05_counts(npred, tp, ntrue):
    """Vectorised per-entity F0.5 (singletons: 1 iff nothing predicted)."""
    npred, tp, ntrue = (np.asarray(x, np.float64) for x in (npred, tp, ntrue))
    p = np.divide(tp, npred, out=np.zeros_like(tp), where=npred > 0)
    r = np.divide(tp, ntrue, out=np.zeros_like(tp), where=ntrue > 0)
    den = 0.25 * p + r
    f = np.divide(1.25 * p * r, den, out=np.zeros_like(tp), where=den > 0)
    return np.where(ntrue == 0, (npred == 0).astype(np.float64), f)


def decide_mask(code, p, t, alpha):
    """Keep candidates with prob >= t and prob >= alpha * best prob of that S1."""
    pmax = np.zeros(code.max() + 1 if len(code) else 0)
    np.maximum.at(pmax, code, p)
    return (p >= t) & (p >= alpha * pmax[code])


def resolve_conflicts(code, eid, p, sim, keep):
    """One-owner post-processing: every S2/S3 record belongs to at most one S1
    entity, so a record predicted for several S1 entities is kept only for the
    one with the highest probability (ties: higher name+address cosine `sim`,
    then lower S1 index). Only the final matches change, never the candidates.
    Returns the new keep mask and counts for logging."""
    idx = np.where(keep)[0]
    d = pd.DataFrame({"e": eid[idx], "p": p[idx], "sim": sim[idx], "c": code[idx], "i": idx})
    n = d.groupby("e")["i"].transform("size").values
    d = d[n > 1]
    out = keep.copy()
    info = {"conflict_records": int(d["e"].nunique()), "pairs_in_conflict": int(len(d)),
            "dropped": 0, "ties": 0}
    if len(d):
        d = d.sort_values(["e", "p", "sim", "c"], ascending=[True, False, False, True], kind="stable")
        first = ~d["e"].duplicated().values
        top_p = d.groupby("e")["p"].transform("max").values
        info["ties"] = int(d[(d["p"].values == top_p)].groupby("e").size().gt(1).sum())
        loser = d["i"].values[~first]
        out[loser] = False
        info["dropped"] = int(len(loser))
    return out, info


def macro_f05(code, y, keep, ntrue):
    """code: S1 index (0..n-1) per pair; ntrue: #true matches per S1 (all S1)."""
    n = len(ntrue)
    npred = np.bincount(code, weights=keep, minlength=n)
    tp = np.bincount(code, weights=keep & (y == 1), minlength=n)
    return float(f05_counts(npred, tp, ntrue).mean())


# ---------------------------------------------------------------- stage A
def _file_hash(name):
    import hashlib
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), name), "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()[:8]


def pass_a(split, paths, args, work, pool):
    """Block every (bucket, target) chunk (cached in `work`) and featurise it
    (cached in `work/f_<feature set>_<features.py hash>`); returns feature files."""
    fdir = os.path.join(work, f"f_{args.feature_set}_{_file_hash('features.py')}")
    os.makedirs(fdir, exist_ok=True)
    files = []
    buckets = prep.bucket_keys(paths[1])
    if args.cross_country:
        buckets = ["__missing__"]  # "__missing__" bucket compares against everything
    for b in buckets:
        cand_f = {t: os.path.join(work, f"{split}_{b}_{t}_cand.parquet") for t in ("S2", "S3")}
        feat_f = {t: os.path.join(fdir, f"{split}_{b}_{t}.parquet") for t in ("S2", "S3")}
        if all(os.path.exists(f) for f in feat_f.values()) and not args.rebuild:
            files += list(feat_f.values())
            log(f"  [{split}/{b}] cached")
            continue
        L = prep.read_bucket(paths[1], b)
        targets = {"S2": prep.read_bucket(paths[2], b, with_missing=True),
                   "S3": prep.read_bucket(paths[3], b, with_missing=True)}
        log(f"  [{split}/{b}] S1={L.num_rows} S2={targets['S2'].num_rows} "
            f"S3={targets['S3'].num_rows}")
        if all(os.path.exists(f) for f in cand_f.values()) and not args.rebuild:
            cands = {t: pd.read_parquet(f) for t, f in cand_f.items()}
            log("    blocking cached")
        else:
            cands = generate_candidates(L, targets, pool, args.k_key, args.key_max_df,
                                        args.store_per_source, args.min_score,
                                        prune_score=args.prune_score,
                                        num_variants=args.num_variants,
                                        keep_addr=args.store_addr, log=log)
            for t, c in cands.items():
                c.to_parquet(cand_f[t], index=False)
        gc.collect()
        tok = BucketTokens(pool, {"S1": L, **targets})
        s1_rows, s1_ids = prep.col(L, "row"), prep.col(L, "entity_id")
        for tag, c in cands.items():
            R = targets[tag]
            ctx = record_context(L, R, L) if args.feature_set == "v3" else None
            bf = base_features(c, L, R, tok.mats["S1"], tok.mats[tag], tok.idf,
                               feature_set=args.feature_set, ctx=ctx)
            bf["cos_key"] = c["cos_key"].values
            if "addr_rank" in c:
                bf["addr_rank"] = c["addr_rank"].values.astype(np.float32)
            meta = pd.DataFrame({
                "s1_row": s1_rows[c["i1_local"].values],
                "i2": prep.col(R, "row", c["i2_local"].values),
                "src": tag,
                "entity_id": prep.col(R, "entity_id", c["i2_local"].values),
                "s1_id": s1_ids[c["i1_local"].values],
                "bucket": b,
            })
            df = pd.concat([meta, bf], axis=1)
            df.to_parquet(feat_f[tag], index=False)
            log(f"    {tag}: {len(df)} pairs ({len(df) / max(1, L.num_rows):.2f}/S1)")
            files.append(feat_f[tag])
            del bf, meta, df
        del L, targets, tok, cands
        gc.collect()
    return files


def keep_mask(df, kps, kpa=0):
    """Pruned candidate list: top-`kps` by the pruning score, plus (address
    channel) the top-`kpa` by address cosine."""
    m = df["cheap_rank"].values < kps
    if kpa and "addr_rank" in df:
        m |= df["addr_rank"].values < kpa
    return m


def load_chunk(f, kps, columns=None, with_c1=False, kpa=0):
    """Read one chunk, keep the pruned candidate list (see keep_mask) and
    optionally add the stage-1 cheap features, which depend on the pruned set
    and are therefore computed here."""
    df = pd.read_parquet(f, columns=columns)
    emb = f.replace(".parquet", "_emb.parquet")
    if USE_EMB and os.path.exists(emb):  # multilingual name-embedding cosine (embed_feature.py)
        df["n_emb_cos"] = pd.read_parquet(emb)["n_emb_cos"].values
    df = df[keep_mask(df, kps, kpa)].reset_index(drop=True)
    if with_c1:
        df = df.drop(columns=[c for c in df.columns if c.startswith("c1_")])
        cf = cheap_features(df, "s1_row", "i2")
        df = pd.concat([df, cf.drop(columns=[c for c in cf.columns if c in df.columns])
                        .add_prefix("c1_")], axis=1)
    return df


# ------------------------------------------------------------------ model
def _rows(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else np.asarray(X[idx])


def oof_lgb(X, y, groups, folds, params, rounds=3000, feature_names="auto"):
    oof = np.zeros(len(X))
    iters = []
    for a, b in GroupKFold(n_splits=folds).split(np.zeros(len(y)), y, groups):
        mdl = lgb.train(params, lgb.Dataset(_rows(X, a), y[a], feature_name=feature_names), rounds,
                        valid_sets=[lgb.Dataset(_rows(X, b), y[b], feature_name=feature_names)],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[b] = mdl.predict(_rows(X, b), num_iteration=mdl.best_iteration)
        iters.append(mdl.best_iteration)
    final = lgb.train(params, lgb.Dataset(X if hasattr(X, "iloc") else np.asarray(X), y,
                                          feature_name=feature_names),
                      int(np.mean(iters) * 1.1) + 1)
    return oof, final, iters


XGB_PARAMS = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                  device="cuda", grow_policy="lossguide", max_depth=0, subsample=0.8,
                  colsample_bytree=0.8, reg_lambda=1.0, min_child_weight=1.0, max_bin=256,
                  seed=SEED)


class XGBModel:
    """Thin wrapper so an XGBoost booster predicts like a LightGBM one."""

    def __init__(self, booster, names):
        self.booster, self.names = booster, names

    def predict(self, X):
        import xgboost as xgb
        return self.booster.predict(xgb.DMatrix(np.asarray(X, dtype=np.float32),
                                                feature_names=self.names))

    def save_model(self, path):
        self.booster.save_model(path)

    def feature_importance(self, kind="gain"):
        sc = self.booster.get_score(importance_type="gain")
        return np.array([sc.get(n, 0.0) for n in self.names])


def oof_xgb(X, y, groups, folds, lr, leaves, rounds, names):
    """Same protocol as oof_lgb (GroupKFold, early stopping 100, final model with
    1.1x mean best iteration), XGBoost hist on the GPU."""
    import xgboost as xgb
    params = dict(XGB_PARAMS, eta=lr, max_leaves=leaves)
    oof = np.zeros(len(y))
    iters = []
    for a, b in GroupKFold(n_splits=folds).split(np.zeros(len(y)), y, groups):
        dtr = xgb.QuantileDMatrix(np.asarray(X[a]), y[a], feature_names=names)
        dva = xgb.QuantileDMatrix(np.asarray(X[b]), y[b], ref=dtr, feature_names=names)
        bst = xgb.train(params, dtr, rounds, evals=[(dva, "valid")], early_stopping_rounds=100,
                        verbose_eval=False)
        oof[b] = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
        iters.append(bst.best_iteration + 1)
        del dtr, dva, bst
        gc.collect()
    dall = xgb.QuantileDMatrix(np.asarray(X), y, feature_names=names)
    final = xgb.train(params, dall, int(np.mean(iters) * 1.1) + 1)
    return oof, XGBModel(final, names), iters


def stage2_frame(df, p1, key_feature=False):
    """Filtered pairs of one chunk -> base + context features (+ stage-1 prob)."""
    drop = META + [c for c in df.columns if c.startswith("c1_")]
    if not key_feature:
        drop.append("cos_key")
    feat = df.drop(columns=[c for c in drop if c in df.columns])
    X = context_features(feat, df["s1_row"].values, df["src"].values, df["i2"].values)
    X["p1"] = p1
    return X


def write_lists(path, s1_ids, mapping, col):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"source1_entity_id\t{col}\n")
        for k in s1_ids:
            fh.write(f"{k}\t{','.join(sorted(mapping.get(k, ())))}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--k-key", type=int, default=30,
                    help="neighbours retrieved per S1 record and source")
    ap.add_argument("--key-max-df", type=float, default=0.0005,
                    help="drop blocking keys shared by more than this share of the bucket")
    ap.add_argument("--prune-score", default="cheap", choices=["cheap", "cos_key", "blend"],
                    help="score used to keep the top --keep-per-source candidates")
    ap.add_argument("--key-feature", action=argparse.BooleanOptionalAction, default=True,
                    help="use the retrieval cosine cos_key as a stage-1/stage-2 feature")
    ap.add_argument("--num-variants", action="store_true",
                    help="house-number blocking keys robust to a dropped first/last digit")
    ap.add_argument("--feature-set", default="v3", choices=["v1", "v2", "v3"],
                    help="v1 = original features, v2 = + house-number/name-rarity/script "
                         "features, v3 = + name-uniqueness and legal-form relation features")
    ap.add_argument("--lr", type=float, default=0.1, help="stage-2 LightGBM learning rate")
    ap.add_argument("--num-leaves", type=int, default=127)
    ap.add_argument("--max-rounds", type=int, default=3000)
    ap.add_argument("--keep-per-source", type=int, default=10,
                    help="candidates kept per S1 record and source (by --prune-score)")
    ap.add_argument("--store-per-source", type=int, default=10,
                    help="candidates featurised and cached per S1 record and source "
                         "(upper bound for --keep-per-source)")
    ap.add_argument("--min-score", type=float, default=0.25)
    ap.add_argument("--keep-addr", type=int, default=5,
                    help="address channel: also keep the top-N candidates by address cosine")
    ap.add_argument("--store-addr", type=int, default=0,
                    help="address-channel candidates cached per S1 record and source")
    ap.add_argument("--cross-country", action="store_true",
                    help="do not restrict blocking to same country label")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--stage1-recall", type=float, default=0.985,
                    help="share of blocked true pairs the stage-1 filter must keep")
    ap.add_argument("--stage1-frac", type=float, default=0.3,
                    help="share of train S1 used to fit the stage-1 filter")
    ap.add_argument("--emb-feature", action="store_true",
                    help="add the multilingual name-embedding cosine (run embed_feature.py first)")
    ap.add_argument("--learner", default="xgb", choices=["lgb", "xgb"],
                    help="stage-2 learner: LightGBM (CPU) or XGBoost hist on the GPU (CUDA)")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="share of train S1 entities used to fit/evaluate the models "
                         "(blocking and recall always use all of them)")
    ap.add_argument("--rebuild", action="store_true", help="ignore cached chunk files")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--resolve-conflicts", action=argparse.BooleanOptionalAction, default=True,
                    help="keep each S2/S3 record only for the S1 with the highest probability "
                         "(matching_results.tsv only; candidates are unchanged)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)
    cache = os.path.join(args.artifacts_dir, "cache")
    args.store_per_source = max(args.store_per_source, args.keep_per_source)
    args.store_addr = max(args.store_addr, args.keep_addr)
    tag = (f"n{prep._rules_hash()}_k{args.k_key}_df{args.key_max_df}_st{args.store_per_source}"
           f"_ms{args.min_score}_{args.prune_score}" + ("_nv" if args.num_variants else "")
           + (f"_ad{args.store_addr}" if args.store_addr else "")

           + ("_xc" if args.cross_country else ""))
    work = os.path.join(args.artifacts_dir, "work", tag)
    os.makedirs(work, exist_ok=True)
    report = {"args": vars(args)}
    global USE_EMB
    USE_EMB = args.emb_feature
    pool = Pool(args.workers)
    if args.key_feature:
        STAGE1_BASE.append("cos_key")
    if args.keep_addr:
        STAGE1_BASE.append("addr_rank")

    # ================================================================ TRAIN
    log("normalising train (cached)")
    paths = {i: prep.ensure_cache(os.path.join(args.data_dir, "train", f"train_source{i}.tsv"),
                                  cache, pool=pool) for i in (1, 2, 3)}
    gt = prep.read_tsv(os.path.join(args.data_dir, "train", "train_ground_truth.tsv"))
    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")
    s1_all = prep.read_bucket(paths[1], "__missing__")
    s1_ids = prep.col(s1_all, "entity_id")
    s1_ck = prep.col(s1_all, "ckey")
    del s1_all
    ntrue_map = dict(zip(gt["source1_entity_id"],
                         gt["matched_entity_ids"].map(lambda s: len([x for x in s.split(",") if x]))))
    ntrue = np.array([ntrue_map.get(k, 0) for k in s1_ids], np.float64)
    ex = gt.assign(m=gt["matched_entity_ids"].str.split(",")).explode("m")
    ex = ex[ex["m"].fillna("") != ""]
    true_keys = np.sort(pair_key(ex["source1_entity_id"].values, ex["m"].values))
    del ex
    log(f"train S1={len(s1_ids)} true pairs={len(true_keys)} "
        f"singleton share={np.mean(ntrue == 0):.4f} avg matches/S1={ntrue.mean():.3f}")

    log("pass A (train): blocking + features")
    files = pass_a("train", paths, args, work, pool)

    # ---- stage-1 training view: only pairs of the sampled S1 are held in memory;
    #      every chunk is scored by the stage-1 model on the fly in pass B
    samp1 = in_sample(s1_ids, args.stage1_frac)   # stage-1 fitting / OOF sample
    samp = in_sample(s1_ids, args.train_frac)     # stage-2 fitting / OOF sample
    n_true = len(true_keys)
    n_tgt = sum(pq.ParquetFile(paths[i]).metadata.num_rows for i in (2, 3))
    kps, kpa = args.keep_per_source, args.keep_addr
    tr_parts, tr_y, ys_chunk = [], [], []
    ranks_all, aranks_all, ys_all = [], [], []
    n_pairs = n_pos = 0
    s1_cols = None
    for f in files:
        names = pq.read_schema(f).names
        rc = ["s1_id", "entity_id", "cheap_rank"] + (["addr_rank"] if "addr_rank" in names else [])
        r = pd.read_parquet(f, columns=rc)
        yf = np.isin(pair_key(r["s1_id"].values, r["entity_id"].values), true_keys)
        ranks_all.append(r["cheap_rank"].values.astype(np.int16))
        aranks_all.append(r["addr_rank"].values.astype(np.int16) if "addr_rank" in r
                          else np.full(len(r), 999, np.int16))
        ys_all.append(yf)
        yk = yf[keep_mask(r, kps, kpa)].astype(np.int8)
        del r
        d = load_chunk(f, kps, list(dict.fromkeys(
            ["s1_row", "src", "i2"] + STAGE1_BASE + (["addr_rank"] if "addr_rank" in names else []))),
            with_c1=True, kpa=kpa)
        s1_cols = s1_cols or (STAGE1_BASE + [c for c in d.columns if c.startswith("c1_")])
        sm = samp1[d["s1_row"].values]
        tr_parts.append(d.loc[sm, s1_cols + ["s1_row"]].reset_index(drop=True))
        tr_y.append(yk[sm])
        ys_chunk.append(yk)
        n_pairs += len(d)
        n_pos += int(yk.sum())
        del d
        gc.collect()
    ranks_all, aranks_all, ys_all = (np.concatenate(ranks_all), np.concatenate(aranks_all),
                                     np.concatenate(ys_all))
    for kk in range(1, int(ranks_all[ranks_all < 999].max()) + 2):
        m = (ranks_all < kk) | (aranks_all < kpa)
        log(f"  keep_per_source={kk} (+addr {kpa}): recall={ys_all[m].sum() / n_true:.4f} "
            f"avg_cands={m.sum() / len(s1_ids):.2f}")
    del ranks_all, aranks_all, ys_all
    report["blocking_train"] = {
        "pairs": int(n_pairs),
        "avg_candidates_per_s1": round(n_pairs / len(s1_ids), 2),
        "pair_recall": round(n_pos / n_true, 4),
        "reduction_ratio": round(1 - n_pairs / (len(s1_ids) * n_tgt), 8),
    }
    log("blocking:", report["blocking_train"])

    # ---- stage-1 filter: cheap vectorised model shrinks the candidate list
    log("stage-1 filter")
    T = pd.concat(tr_parts, ignore_index=True)
    tr_lens = [len(t) for t in tr_parts]
    del tr_parts
    y_tr = np.concatenate(tr_y).astype(int)
    p1_s, stage1, _ = oof_lgb(T[s1_cols], y_tr, T["s1_row"].values, args.folds,
                              dict(LGB_PARAMS, num_leaves=15, learning_rate=0.1), 500)
    del T
    gc.collect()
    pos = np.sort(p1_s[y_tr == 1])
    t1 = float(pos[int((1 - args.stage1_recall) * len(pos))]) if len(pos) else 0.0

    # ---- pass B: stage-1 filter + context features, keep sampled S1 for stage 2
    log("pass B (train): filter + context features")
    ys, codes, metas, subm, ekeys, sims = [], [], [], [], [], []
    xpath = os.path.join(args.artifacts_dir, "stage2_X.f32")
    xfh = open(xpath, "wb")
    n_rows, feat_cols = 0, None
    sub01 = in_sample(s1_ids, min(args.stage1_frac, args.train_frac))  # comparable subset
    n_kept = n_kept_pos = 0
    tp_s1 = np.zeros(len(s1_ids))
    off = 0
    for f, yk, nl in zip(files, ys_chunk, tr_lens):
        df = load_chunk(f, kps, with_c1=True, kpa=kpa)
        sm = samp1[df["s1_row"].values]
        p1 = np.empty(len(df))
        p1[sm] = p1_s[off:off + nl]  # out-of-fold for the stage-1 sample
        off += nl
        if (~sm).any():
            p1[~sm] = stage1.predict(df.loc[~sm, s1_cols])
        k = p1 >= t1
        n_kept += int(k.sum())
        n_kept_pos += int(yk[k].sum())
        tp_s1 += np.bincount(df["s1_row"].values[k], weights=yk[k], minlength=len(s1_ids))
        df = df[k].reset_index(drop=True)
        X = stage2_frame(df, p1[k], args.key_feature)
        s = samp[df["s1_row"].values]
        feat_cols = feat_cols or list(X.columns)
        xfh.write(np.ascontiguousarray(X.loc[s, feat_cols].to_numpy(np.float32)).tobytes())
        n_rows += int(s.sum())
        ys.append(yk[k][s])
        codes.append(df["s1_row"].values[s])
        sb = sub01[df["s1_row"].values[s]]
        subm.append(sb)
        metas.append(df.loc[s, ["s1_id", "entity_id", "bucket"]].reset_index(drop=True)[sb]
                     .reset_index(drop=True))
        ekeys.append(np.where(df["src"].values[s] == "S3", 2, 1).astype(np.int64) * 10 ** 10
                     + id_num(df["entity_id"].values[s]))
        sims.append((df["cos_name"].values[s] + df["cos_addr"].values[s]).astype(np.float32))
        del df, X
        gc.collect()
    xfh.close()
    report["stage1"] = {"threshold": t1,
                        "avg_candidates_per_s1": round(n_kept / len(s1_ids), 2),
                        "pair_recall": round(n_kept_pos / n_true, 4),
                        "blocking_ceiling_f05_all_s1": round(
                            float(f05_counts(tp_s1, tp_s1, ntrue).mean()), 4)}
    log("after stage-1:", report["stage1"])
    X = np.memmap(xpath, dtype=np.float32, mode="r", shape=(n_rows, len(feat_cols)))
    y = np.concatenate(ys)
    code = np.concatenate(codes)
    sub_mask = np.concatenate(subm)
    ekey = np.concatenate(ekeys)
    simv = np.concatenate(sims)
    meta = pd.concat(metas, ignore_index=True)
    del ys, codes, metas, subm, ekeys, sims

    # evaluation universe: all sampled S1 (incl. those without any candidate)
    samp_idx = np.where(samp)[0]
    remap = np.full(len(s1_ids), -1)
    remap[samp_idx] = np.arange(len(samp_idx))
    ecode = remap[code]
    entrue = ntrue[samp_idx]

    log(f"{args.folds}-fold OOF training on {len(X)} pairs of {len(samp_idx)} S1, "
        f"positives={y.sum()}")
    if args.learner == "xgb":
        oof, final, best_iters = oof_xgb(X, y, code, args.folds, args.lr, args.num_leaves,
                                         args.max_rounds, feat_cols)
    else:
        params2 = dict(LGB_PARAMS, learning_rate=args.lr, num_leaves=args.num_leaves)
        oof, final, best_iters = oof_lgb(X, y, code, args.folds, params2, args.max_rounds,
                                         feature_names=feat_cols)
    log("  best iters:", best_iters)

    best = (-1, None, None)
    for t in np.arange(0.20, 0.96, 0.025):
        for alpha in (0.0, 0.3, 0.5, 0.7):
            s = macro_f05(ecode, y, decide_mask(ecode, oof, t, alpha), entrue)
            if s > best[0]:
                best = (s, float(t), alpha)
    score, thr, alpha = best
    tp_e = np.bincount(ecode, weights=y, minlength=len(samp_idx))
    ceiling = float(f05_counts(tp_e, tp_e, entrue).mean())
    report["validation"] = {"oof_macro_f05": round(score, 4), "threshold": thr, "alpha": alpha,
                            "blocking_ceiling_f05": round(ceiling, 4),
                            "all_empty_baseline": round(float((entrue == 0).mean()), 4),
                            "n_s1_evaluated": int(len(samp_idx))}
    keep = decide_mask(ecode, oof, thr, alpha)
    # comparable score on the fixed 30 % subset (samples are nested), and with
    # one-owner conflict resolution among the evaluated S1
    sub_idx = np.where(sub01 & samp)[0]
    sremap = np.full(len(s1_ids), -1)
    sremap[sub_idx] = np.arange(len(sub_idx))
    report["validation"]["oof_macro_f05_subset30"] = round(
        macro_f05(sremap[code[sub_mask]], y[sub_mask], keep[sub_mask], ntrue[sub_idx]), 5)
    kres, _ = resolve_conflicts(ecode, ekey, oof, simv, keep)
    report["validation"]["oof_macro_f05_resolved"] = round(macro_f05(ecode, y, kres, entrue), 5)
    report["validation"]["oof_macro_f05_resolved_subset30"] = round(
        macro_f05(sremap[code[sub_mask]], y[sub_mask], kres[sub_mask], ntrue[sub_idx]), 5)
    log("validation:", report["validation"])
    per_c = {}
    ck_s = s1_ck[samp_idx]
    for c in sorted(set(ck_s)):
        m = ck_s == c
        sub = np.where(m)[0]
        mp = np.full(len(samp_idx), -1)
        mp[sub] = np.arange(len(sub))
        pm = m[ecode]
        per_c[c] = round(macro_f05(mp[ecode[pm]], y[pm], keep[pm], entrue[sub]), 4)
    report["validation"]["per_country"] = per_c
    log("per-country OOF F0.5:", per_c)

    # OOF pairs for error analysis
    oo = meta.assign(y=y[sub_mask], p=oof[sub_mask], pred=keep[sub_mask])
    for c in ("cos_name", "cos_addr", "n_tset", "a_tset", "cheap", "cheap_rank", "addr_rank", "p1"):
        if c in feat_cols:
            oo[c] = np.asarray(X[:, feat_cols.index(c)])[sub_mask]
    oo.to_parquet(os.path.join(args.artifacts_dir, "oof_pairs.parquet"), index=False)

    final.save_model(os.path.join(args.artifacts_dir, "model_stage2.txt" if args.learner == "lgb"
                                  else "model_stage2_xgb.json"))
    stage1.save_model(os.path.join(args.artifacts_dir, "model_stage1.txt"))
    imp = pd.Series(final.feature_importance("gain"), index=feat_cols).sort_values(ascending=False)
    report["top_features"] = imp.head(15).round(1).to_dict()
    with open(os.path.join(args.artifacts_dir, "run_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    del X, oof, meta, oo
    gc.collect()
    os.remove(xpath)

    # ================================================================= TEST
    if not args.skip_test:
        log("normalising test (cached)")
        tpaths = {i: prep.ensure_cache(os.path.join(args.data_dir, "test", f"test_source{i}.tsv"),
                                       cache, pool=pool) for i in (1, 2, 3)}
        ids_t = prep.col(prep.read_bucket(tpaths[1], "__missing__"), "entity_id")
        tcode = {k: i for i, k in enumerate(ids_t)}
        log("pass A (test): blocking + features")
        tfiles = pass_a("test", tpaths, args, work, pool)
        codes, eids, probs, sims, n_blocked = [], [], [], [], 0
        for f in tfiles:
            df = load_chunk(f, args.keep_per_source, with_c1=True, kpa=args.keep_addr)
            n_blocked += len(df)
            q1 = stage1.predict(df[s1_cols])
            k = q1 >= t1
            df = df[k].reset_index(drop=True)
            Xt = stage2_frame(df, q1[k], args.key_feature)[feat_cols]
            probs.append(final.predict(Xt))
            codes.append(df["s1_id"].map(tcode).values)
            eids.append(df["entity_id"].values)
            sims.append((df["cos_name"] + df["cos_addr"]).values)
            del df, Xt
            gc.collect()
        tc = np.concatenate(codes)
        te = np.concatenate(eids)
        pt = np.concatenate(probs)
        km = decide_mask(tc, pt, thr, alpha)
        if args.resolve_conflicts:
            km, cinfo = resolve_conflicts(tc, te, pt, np.concatenate(sims), km)
            report["test_conflicts"] = cinfo
            log("conflict resolution:", cinfo)
        cmap_t, pred_t = {}, {}
        for c_, e_, k_ in zip(tc, te, km):
            cmap_t.setdefault(ids_t[c_], set()).add(e_)
            if k_:
                pred_t.setdefault(ids_t[c_], set()).add(e_)
        assert all(pred_t[k] <= cmap_t[k] for k in pred_t)
        write_lists(os.path.join(args.out_dir, "candidate_pairs.tsv"), ids_t, cmap_t,
                    "candidate_entity_ids")
        write_lists(os.path.join(args.out_dir, "matching_results.tsv"), ids_t, pred_t,
                    "matched_entity_ids")
        npred = np.array([len(pred_t.get(k, ())) for k in ids_t])
        t_ck = prep.col(prep.read_bucket(tpaths[1], "__missing__"), "ckey")
        report["test"] = {
            "s1": len(ids_t),
            "avg_blocked_per_s1": round(n_blocked / len(ids_t), 2),
            "avg_candidates_per_s1": round(len(tc) / len(ids_t), 2),
            "predicted_singletons": round(float(np.mean(npred == 0)), 4),
            "avg_matches_per_s1": round(float(npred.mean()), 3),
            "countries": pd.Series(t_ck).value_counts().to_dict(),
            "predicted_singletons_per_country": {
                c: round(float(np.mean(npred[t_ck == c] == 0)), 4) for c in sorted(set(t_ck))},
        }
        log("test:", report["test"])
    with open(os.path.join(args.artifacts_dir, "run_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    pool.close()
    log("done ->", args.out_dir)


if __name__ == "__main__":
    main()
