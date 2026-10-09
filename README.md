# Amazon ML Challenge 2026 — Business Entity Resolution (Team TangoAlpha)

Match business records from three noisy sources (US, India, and France in the
test set): for every Source-1 entity, find all Source-2 / Source-3 records of
the same real-world business. Scored by macro F0.5 per Source-1 entity.

**Result:** out-of-fold macro F0.5 **0.9704** on 2.2M training entities
(baseline 0.9383), blocking ceiling 0.9805, ~5 candidates per entity.

## Approach

1. **Normalisation** — transliteration of Indic scripts, canonical legal forms
   (including transliterated ones such as `praaivett limittedd` → pvt ltd, matched
   by consonant skeleton), abbreviations, leetspeak, French street types, house
   numbers.
2. **Blocking** — IDF-weighted sparse top-k retrieval over discrete name and
   address keys (multi-threaded, ~1 min per million queries), exact char n-gram
   cosines, pruning by a blended score **plus** an address channel that rescues
   coined brand names, web-domain names and transliterated names at the same
   address.
3. **Stage-1 filter** — LightGBM on cheap scores cuts the list to ~4–5
   candidates per entity (98.5 % of blocked true pairs kept).
4. **Stage-2 matcher** — XGBoost on the GPU, ~80 name / address / house-number /
   name-uniqueness / legal-form / context features, trained on all training
   entities.
5. **Decision** — threshold tuned for macro F0.5 on grouped out-of-fold
   predictions, then one-owner conflict resolution (each Source-2/3 record goes
   to at most one Source-1 entity).

Every change was judged by out-of-fold macro F0.5 (GroupKFold by Source-1 id);
the full log of kept and rejected experiments is in
[`results/experiments.md`](results/experiments.md), per-run reports in
[`results/runs/`](results/runs/), and the write-up in
[`Documentation_template.md`](Documentation_template.md).

## Repository layout

```
code/business_entity_resolution/   pipeline (src/), README with run instructions, requirements.txt
Documentation_template.md          methodology write-up
results/                           experiment log and run reports
```

## Running it

The competition data is not included. Place it as `dataset/train` and
`dataset/test` next to `code/` and follow
[`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md):

```bash
python -m pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/pipeline.py --data-dir dataset --out-dir output --artifacts-dir artifacts
```

The default stage-2 learner needs a CUDA GPU; on CPU use
`--learner lgb --train-frac 0.3`.

## Team

Ankit Bhaumik, Samarth Agarwal, Bhavneet Dhawan, Shivam Rajhuvanshi.
