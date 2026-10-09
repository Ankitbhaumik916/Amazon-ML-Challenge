"""Multilingual name-embedding cosine for transliterated candidate pairs.

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0,
118M parameters), run on the GPU. It reads the *original* names, so an Indic-
script name and its English form can be compared directly (e.g. "Ram Marketing
Private Limited" vs "राम मार्केटिंग प्राइवेट लिमिटेड"). To keep it cheap it is
computed only for candidate pairs whose Source-2/3 name is written in a
non-Latin script (all other pairs get NaN), for the pairs already in the cached
candidate chunks. Output: one sidecar parquet per feature chunk
("<chunk>_emb.parquet", column n_emb_cos, same row order), joined at load time.

    python code/business_entity_resolution/src/embed_feature.py --artifacts-dir artifacts
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep  # noqa: E402

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--splits", nargs="*", default=["train", "test"])
    ap.add_argument("--batch", type=int, default=1024)
    a = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    rep = json.load(open(os.path.join(a.artifacts_dir, "run_report.json")))
    args = rep["args"]
    work = [d for d in glob.glob(os.path.join(a.artifacts_dir, "work", "*"))
            if f"_st{args['store_per_source']}_" in d and d.endswith(f"_ad{args['store_addr']}")][0]
    fdir = max(glob.glob(os.path.join(work, f"f_{args['feature_set']}_*")), key=os.path.getmtime)
    model = SentenceTransformer(MODEL, device="cuda")
    model.half()
    cache = os.path.join(a.artifacts_dir, "cache")
    for split in a.splits:
        t0 = time.time()
        files = sorted(glob.glob(os.path.join(fdir, f"{split}_*.parquet")))
        files = [f for f in files if not f.endswith("_emb.parquet")]
        # non-Latin-script target records (flag computed at normalisation time)
        script = {}
        for i in (2, 3):
            cp = prep.ensure_cache(os.path.join(a.data_dir, split, f"{split}_source{i}.tsv"), cache)
            t = pq.read_table(cp, columns=["entity_id", "n_script"]).to_pandas()
            script.update(dict.fromkeys(t.entity_id[t.n_script > 0].values, True))
        need_rows, need_ids = {}, set()
        for f in files:
            d = pd.read_parquet(f, columns=["s1_id", "entity_id"])
            m = d.entity_id.map(script).fillna(False).values.astype(bool)
            need_rows[f] = m
            need_ids |= set(d.s1_id.values[m]) | set(d.entity_id.values[m])
        print(f"[{split}] pairs needing the embedding: {sum(int(m.sum()) for m in need_rows.values())}, "
              f"records: {len(need_ids)}", flush=True)
        names = {}
        for i in (1, 2, 3):
            for ch in pd.read_csv(os.path.join(a.data_dir, split, f"{split}_source{i}.tsv"), sep="\t",
                                  dtype=str, keep_default_na=False, quoting=3, chunksize=1000000,
                                  usecols=["entity_id", "business_name"]):
                ch = ch[ch.entity_id.isin(need_ids)]
                names.update(zip(ch.entity_id, ch.business_name))
        uniq = sorted(set(names.values()))
        pos = {s: k for k, s in enumerate(uniq)}
        emb = np.zeros((len(uniq), model.get_sentence_embedding_dimension()), dtype=np.float16)
        for s in range(0, len(uniq), 200000):  # blocks: never hold all float32 vectors
            emb[s:s + 200000] = model.encode(uniq[s:s + 200000], batch_size=a.batch,
                                             normalize_embeddings=True, convert_to_numpy=True,
                                             show_progress_bar=False).astype(np.float16)
        print(f"[{split}] embedded {len(uniq)} unique names in {time.time() - t0:.0f}s", flush=True)
        for f, m in need_rows.items():
            d = pd.read_parquet(f, columns=["s1_id", "entity_id"])
            out = np.full(len(d), np.nan, dtype=np.float32)
            if m.any():
                ia = np.array([pos[names[x]] for x in d.s1_id.values[m]])
                ib = np.array([pos[names[x]] for x in d.entity_id.values[m]])
                out[m] = np.einsum("ij,ij->i", emb[ia].astype(np.float32), emb[ib].astype(np.float32))
            pd.DataFrame({"n_emb_cos": out}).to_parquet(f.replace(".parquet", "_emb.parquet"), index=False)
        print(f"[{split}] done in {time.time() - t0:.0f}s", flush=True)
        del emb, names, pos, uniq


if __name__ == "__main__":
    main()
