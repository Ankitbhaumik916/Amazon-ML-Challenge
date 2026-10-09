"""Error analysis on the out-of-fold predictions of the last train run.

For a random subset of the evaluated Source-1 entities every true pair is put in
one bucket - missed by blocking (never among the stored top-N per source),
dropped by the stage-1 filter, false negative of the matcher - and every
predicted pair that is wrong is a false positive. Each pair is tagged with rough
patterns (computed on all pairs of the bucket, so the counts are exact for the
subset) and a sample is printed with the raw strings.

Run from student_resource/ after a pipeline run:
    python code/business_entity_resolution/src/error_analysis.py --artifacts-dir artifacts -n 150
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import in_sample  # noqa: E402
from prep import read_tsv  # noqa: E402

LANDMARK = re.compile(r"\b(near|nr|opp|opposite|behind|beside|next to|c/o|c o|landmark)\b", re.I)


def read_ids(path, ids, columns=None):
    t = pq.read_table(path, columns=columns, filters=[("entity_id", "in", list(ids))]) \
        if columns else None
    if t is None:
        tbl = pq.read_table(path)
        t = tbl.filter(pc.is_in(tbl.column("entity_id"), pa.array(list(ids))))
    return t.to_pandas().set_index("entity_id")


def tag_pair(a, b):
    """Rough patterns for a (S1, other) pair; a/b are rows of normalised fields."""
    tags = []
    na, nb = a.n_core, b.n_core
    ts = fuzz.token_set_ratio(na, nb)
    if b.n_script or a.n_script:
        tags.append("translit_name")
    if na == nb:
        if a.n_full != b.n_full:
            tags.append("name_same_core_legal_diff")
        else:
            tags.append("name_identical")
    elif sorted(na.split()) == sorted(nb.split()):
        tags.append("name_word_order")
    elif ts >= 90:
        tags.append("name_typo_or_extra_word")
    elif ts >= 60:
        tags.append("name_partial")
    else:
        tags.append("name_unrelated(<60)")
    if not b.a_full or not a.a_full:
        tags.append("addr_empty_one_side")
    else:
        fa, fb = a.a_first_num, b.a_first_num
        if fa and fb and fa != fb:
            da = "".join(ch for ch in fa if ch.isdigit())
            db = "".join(ch for ch in fb if ch.isdigit())
            if Levenshtein.distance(fa, fb) == 1 and abs(len(fa) - len(fb)) == 1:
                tags.append("num_one_digit_dropped/added")
            elif da == db:
                tags.append("num_letter_suffix")
            elif da and db and Levenshtein.distance(da, db) == 1:
                tags.append("num_one_digit_typo")
            elif da and db and abs(int(da[:9]) - int(db[:9])) <= 30:
                tags.append("num_nearby(<=30)")
            else:
                tags.append("num_other_diff")
        elif bool(fa) != bool(fb):
            tags.append("num_missing_one_side")
        wa, wb = set(a.a_words.split()), set(b.a_words.split())
        if wa and wb and (wa <= wb or wb <= wa) and wa != wb:
            tags.append("addr_components_missing(subset)")
        at = fuzz.token_set_ratio(a.a_full, b.a_full)
        if at < 60:
            tags.append("addr_low_overlap(<60)")
        if a.a_tail != b.a_tail:
            tags.append("addr_tail(city/state)_diff")
        pa_, pb_ = set(a.a_postal.split()), set(b.a_postal.split())
        if pa_ and pb_ and not pa_ & pb_:
            tags.append("postal_diff")
        elif bool(pa_) != bool(pb_):
            tags.append("postal_missing_one_side")
    if a.a_landmark or b.a_landmark:
        tags.append("landmark_words")
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--n-s1", type=int, default=100000, help="S1 subset size")
    ap.add_argument("-n", type=int, default=40, help="examples printed per bucket")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rep = json.load(open(os.path.join(a.artifacts_dir, "run_report.json")))
    oo = pd.read_parquet(os.path.join(a.artifacts_dir, "oof_pairs.parquet"))
    gt = read_tsv(os.path.join(a.data_dir, "train", "train_ground_truth.tsv")).fillna("")
    s1 = gt["source1_entity_id"].values
    samp = s1[in_sample(s1, rep["args"]["train_frac"])]
    rs = np.random.RandomState(a.seed)
    U = set(rs.choice(samp, min(a.n_s1, len(samp)), replace=False))
    oo = oo[oo.s1_id.isin(U)]
    gt = gt[gt.source1_entity_id.isin(U)]
    ex = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    ex = ex[ex.m != ""].rename(columns={"source1_entity_id": "s1_id", "m": "entity_id"})[["s1_id", "entity_id"]]

    # stored (blocked) pairs of the subset: feature chunks of the newest blocking dir
    work = max(glob.glob(os.path.join(a.artifacts_dir, "work", "*")), key=os.path.getmtime)
    fdir = max(glob.glob(os.path.join(work, "f_*")), key=os.path.getmtime)
    kps = rep["args"]["keep_per_source"]
    bl = []
    for f in [x for x in glob.glob(os.path.join(fdir, "train_*.parquet")) if not x.endswith("_emb.parquet")]:
        t = pq.read_table(f, columns=["s1_id", "entity_id", "cheap_rank"])
        t = t.filter(pc.is_in(t.column("s1_id"), pa.array(list(U))))
        d = t.to_pandas()
        bl.append(d[d.cheap_rank < kps])
    bl = pd.concat(bl)
    key = lambda d: set(zip(d.s1_id, d.entity_id))  # noqa: E731
    kb, ko = key(bl), key(oo)
    tp = list(zip(ex.s1_id, ex.entity_id))
    miss_block = [p for p in tp if p not in kb]
    miss_s1 = [p for p in tp if p in kb and p not in ko]
    fn = oo[~oo.pred & (oo.y == 1)]
    fp = oo[oo.pred & (oo.y == 0)]
    n_true = len(tp)
    print(f"subset: {len(U)} S1, {n_true} true pairs, {len(bl)} blocked pairs, {len(oo)} candidate pairs")
    print(f"  missed by blocking : {len(miss_block):6d} ({len(miss_block) / n_true:.2%} of true pairs)")
    print(f"  dropped by stage-1 : {len(miss_s1):6d} ({len(miss_s1) / n_true:.2%})")
    print(f"  FN of the matcher  : {len(fn):6d} ({len(fn) / n_true:.2%})")
    print(f"  FP of the matcher  : {len(fp):6d}")
    buckets = {
        "MISSED BY BLOCKING": pd.DataFrame(miss_block, columns=["s1_id", "entity_id"]),
        "DROPPED BY STAGE-1 FILTER": pd.DataFrame(miss_s1, columns=["s1_id", "entity_id"]),
        "FALSE NEGATIVES (in candidate set)": fn[["s1_id", "entity_id", "p"]],
        "FALSE POSITIVES": fp[["s1_id", "entity_id", "p"]],
    }
    need = set()
    for d in buckets.values():
        need |= set(d.s1_id) | set(d.entity_id)
    cache = os.path.join(a.artifacts_dir, "cache")
    norm = pd.concat([read_ids(p, [x for x in need if x.startswith(f"S{i}-")])
                      for i, p in ((i, max(glob.glob(os.path.join(cache, f"train_source{i}_*.parquet")),
                                               key=os.path.getmtime)) for i in (1, 2, 3))])
    raws = []
    for i in (1, 2, 3):
        for ch in pd.read_csv(os.path.join(a.data_dir, "train", f"train_source{i}.tsv"), sep="\t",
                              dtype=str, keep_default_na=False, quoting=3, chunksize=1000000):
            raws.append(ch[ch.entity_id.isin(need)])
    raw = pd.concat(raws).set_index("entity_id")
    for title, d in buckets.items():
        tags = [tag_pair(norm.loc[s], norm.loc[e]) for s, e in zip(d.s1_id, d.entity_id)]
        cnt = Counter(t for ts in tags for t in ts)
        print(f"\n===== {title}: {len(d)} pairs — pattern counts (a pair can have several)")
        for t, c in cnt.most_common():
            print(f"   {c:6d}  {c / max(1, len(d)):6.1%}  {t}")
        idx = rs.permutation(len(d))[:a.n]
        print(f"----- {len(idx)} examples")
        for j in idx:
            r = d.iloc[j]
            p = f" p={r.p:.2f}" if "p" in d else ""
            print(f"[{'/'.join(tags[j])}]{p}")
            print(f"   S1: {raw.business_name.get(r.s1_id)!s:48.48} | {raw.business_address.get(r.s1_id)!s:.95}")
            print(f"   {r.entity_id[:2]}: {raw.business_name.get(r.entity_id)!s:48.48} | "
                  f"{raw.business_address.get(r.entity_id)!s:.95}")


if __name__ == "__main__":
    main()
