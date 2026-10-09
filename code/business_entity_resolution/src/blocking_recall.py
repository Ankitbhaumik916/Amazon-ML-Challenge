"""Cheap blocking-recall check: retrieval + pruning for a sample of train S1
queries against the full targets of their country bucket, no models.

Reports, per stage, the share of true pairs reached, and splits the misses by
simple patterns. Used to verify a blocking/normalisation change before a full
pipeline run.

    python code/business_entity_resolution/src/blocking_recall.py --n-query 40000
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep  # noqa: E402
from blocking import generate_candidates  # noqa: E402
from pipeline import pair_key  # noqa: E402


def report_patterns(L, targets, ids1, gt, got_ret, got_kept, rs):
    from collections import Counter
    from error_analysis import tag_pair
    g = gt[gt.source1_entity_id.isin(set(ids1))]
    tp = [(s, m) for s, v in zip(g.source1_entity_id, g.matched_entity_ids) for m in v.split(",") if m]
    miss_r = [p for p in tp if p not in got_ret]
    miss_p = [p for p in tp if p in got_ret and p not in got_kept]
    frames = [L.to_pandas()] + [t.to_pandas() for t in targets.values()]
    norm = pd.concat(frames).drop_duplicates("entity_id").set_index("entity_id")
    for name, lst in (("RETRIEVAL MISS", miss_r), ("PRUNED AWAY", miss_p)):
        cnt = Counter(t for s, e in lst for t in tag_pair(norm.loc[s], norm.loc[e]))
        print(f"   {name}: {len(lst)} pairs ({len(lst) / max(1, len(tp)):.2%} of true)")
        for t, c in cnt.most_common(12):
            print(f"      {c:6d} {c / max(1, len(lst)):6.1%}  {t}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--n-query", type=int, default=40000, help="S1 queries per bucket")
    ap.add_argument("--k-key", type=int, default=30)
    ap.add_argument("--key-max-df", type=float, default=0.0005)
    ap.add_argument("--keep", type=int, default=10)
    ap.add_argument("--prune-score", default="cheap")
    ap.add_argument("--num-variants", action="store_true")
    ap.add_argument("--keep-addr", type=int, nargs="*", default=[0])
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--tag", default="")
    ap.add_argument("--patterns", action="store_true", help="tag missed pairs by pattern and stage")
    a = ap.parse_args()
    pool = Pool(a.workers)
    cache = os.path.join(a.artifacts_dir, "cache")
    paths = {i: prep.ensure_cache(os.path.join(a.data_dir, "train", f"train_source{i}.tsv"), cache,
                                  pool=pool) for i in (1, 2, 3)}
    gt = prep.read_tsv(os.path.join(a.data_dir, "train", "train_ground_truth.tsv")).fillna("")
    ex = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    ex = ex[ex.m != ""]
    true_keys = np.sort(pair_key(ex.source1_entity_id.values, ex.m.values))
    ntrue = dict(zip(gt.source1_entity_id, gt.matched_entity_ids.map(lambda s: len([x for x in s.split(",") if x]))))
    rs = np.random.RandomState(0)
    tot = {"true": 0, "retrieved": 0, "kept": 0, "pairs_kept": 0, "q": 0}
    for b in prep.bucket_keys(paths[1]):
        t0 = time.time()
        L = prep.read_bucket(paths[1], b)
        L = L.take(np.sort(rs.choice(L.num_rows, min(a.n_query, L.num_rows), replace=False)))
        targets = {"S2": prep.read_bucket(paths[2], b, with_missing=True),
                   "S3": prep.read_bucket(paths[3], b, with_missing=True)}
        # retrieval-only view: keep everything retrieved
        cands = generate_candidates(L, targets, pool, a.k_key, a.key_max_df, 10 ** 6, -1.0,
                                    prune_score=a.prune_score, num_variants=a.num_variants,
                                    keep_addr=max(a.keep_addr),
                                    log=lambda *x: None)
        ids1 = prep.col(L, "entity_id")
        n_true = sum(ntrue.get(k, 0) for k in ids1)
        ret = kept = pk = 0
        got_ret, got_kept = set(), set()
        kept_by, pk_by = {}, {}
        for tag, c in cands.items():
            e = prep.col(targets[tag], "entity_id", c["i2_local"].values)
            s1e = ids1[c["i1_local"].values]
            y = np.isin(pair_key(s1e, e), true_keys)
            m = (c["cheap_rank"].values < a.keep) & (c["cheap"].values >= 0.25)
            ret += y.sum()
            for ka in a.keep_addr:
                mk = m | (c["addr_rank"].values < ka)
                kept_by[ka] = kept_by.get(ka, 0) + y[mk].sum()
                pk_by[ka] = pk_by.get(ka, 0) + mk.sum()
            m = m | (c["addr_rank"].values < a.keep_addr[-1])
            kept += y[m].sum()
            pk += m.sum()
            if a.patterns:
                got_ret |= set(zip(s1e[y], e[y]))
                got_kept |= set(zip(s1e[y & m], e[y & m]))
        if a.patterns:
            report_patterns(L, targets, ids1, gt, got_ret, got_kept, rs)
        for ka in a.keep_addr:
            print(f"   keep {a.keep} + addr {ka}: recall={kept_by[ka] / n_true:.4f} pairs/S1={pk_by[ka] / L.num_rows:.2f}")
            tot[f"k{ka}"] = tot.get(f"k{ka}", 0) + kept_by[ka]
            tot[f"p{ka}"] = tot.get(f"p{ka}", 0) + pk_by[ka]
        print(f"[{b}] q={L.num_rows} true={n_true} retrieval@{a.k_key}={ret / n_true:.4f} "
              f"kept@{a.keep}={kept / n_true:.4f} kept pairs/S1={pk / L.num_rows:.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        for k, v in (("true", n_true), ("retrieved", ret), ("kept", kept), ("pairs_kept", pk), ("q", L.num_rows)):
            tot[k] += v
    print(f"ALL{(' ' + a.tag) if a.tag else ''}: retrieval@{a.k_key}={tot['retrieved'] / tot['true']:.4f} "
          f"kept@{a.keep}={tot['kept'] / tot['true']:.4f} kept pairs/S1={tot['pairs_kept'] / tot['q']:.2f}")
    for ka in a.keep_addr:
        print(f"ALL keep {a.keep} + addr {ka}: recall={tot[f'k{ka}'] / tot['true']:.4f} "
              f"pairs/S1={tot[f'p{ka}'] / tot['q']:.2f}")
    pool.close()


if __name__ == "__main__":
    main()
