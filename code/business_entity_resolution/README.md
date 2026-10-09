# Business Entity Resolution — Amazon ML Challenge 2026

Key-based sparse blocking + learned stage-1 filter + gradient-boosted pairwise
matcher. Regenerates both `output/candidate_pairs.tsv` and
`output/matching_results.tsv` from the raw train/test TSVs. Only the provided
data is used — no external data, APIs, geocoding or pretrained models.
Models: XGBoost stage-2 matcher (Apache-2.0) and LightGBM stage-1 filter (MIT), a few MB
each — far below 8B parameters. Needs a CUDA GPU for the default XGBoost learner.

## Layout expected

```
student_resource/
├── dataset/train/  train_source{1,2,3}.tsv, train_ground_truth.tsv
├── dataset/test/   test_source{1,2,3}.tsv
├── utils/validate_submission.py
├── Documentation_template.md
└── code/business_entity_resolution/   <- this folder
    ├── src/  pipeline.py      end-to-end entry point
    │         prep.py          loading + parallel normalisation (parquet cache)
    │         normalize.py     name/address normalisation rules
    │         blocking.py      key-based sparse top-k retrieval, exact cosines, pruning
    │         features.py      vectorised pair features + context features
    │         error_analysis.py   OOF errors + blocking misses, tagged by pattern
    │         blocking_recall.py  cheap blocking-recall check (no models)
    │         tune_candidates.py  pre-screen --keep-per-source / --keep-addr / --stage1-recall
    │         conflict_eval.py    evaluate one-owner conflict resolution on train
    │         decision.py / decision_eval.py  expected-F0.5 decision rule (tried, rejected)
    │         test_diag.py        label-free per-country diagnostics of the test predictions
    │         embed_feature.py    multilingual name-embedding cosine (tried, rejected)
    │         make_submission.py  builds <team>_submission.zip
    ├── requirements.txt  run.sh  run.bat
```

## Reproduce end-to-end

```bash
cd student_resource
python -m pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/pipeline.py \
    --data-dir dataset --out-dir output --artifacts-dir artifacts
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
(or just `code/business_entity_resolution/run.sh` / `run.bat`). The defaults
are the final configuration; no flags are needed.

One run does:
1. **normalise** all six source files in parallel (cached as parquet under
   `artifacts/cache/`, keyed by a hash of `normalize.py`);
2. **block** every (country bucket × target source) chunk: sparse TF-IDF over
   discrete keys (name tokens, phonetic skeletons, name bigrams, address words,
   house numbers, postal codes, address bigrams), exact top-30 per source with a
   multi-threaded sparse top-k product, exact char n-gram cosines, keep the top
   10 per source by a blended name/address score **plus** the top 5 by address
   cosine (address channel: coined/brand, web-domain and transliterated names at
   the same address); then compute the pair features (cached under `artifacts/work/`);
3. train the **stage-1 filter** (grouped OOF) and cut the candidate list at the
   threshold keeping 98.5 % of blocked true pairs → this list *is*
   `candidate_pairs.tsv`;
4. train the **stage-2 matcher** — XGBoost `hist` on the GPU (CUDA), trained on
   *all* training Source-1 entities — with 5-fold GroupKFold (grouped by Source-1
   id) and tune the decision threshold for macro F0.5 on the out-of-fold predictions;
5. retrain on all training pairs, then block → filter → score the test set;
6. **one-owner conflict resolution**: an S2/S3 record predicted for several S1
   entities is kept only for the one with the highest probability (ties: higher
   name+address cosine) — `matching_results.tsv` only, the candidate list is
   not pruned — and write both TSVs.

Diagnostics go to `artifacts/run_report.json` (blocking recall, candidates per
entity, OOF F0.5 overall and per country, blocking ceiling, feature importance),
the out-of-fold pairs to `artifacts/oof_pairs.parquet`, the models to
`artifacts/model_stage{1,2}.txt`.

The stage-1 filter is fitted on a deterministic 30 % sample of the training
Source-1 entities (`--stage1-frac 0.3`); the stage-2 matcher uses all of them
(`--train-frac 1.0`, 2.2M entities, 9.4M pairs), streamed to a float32 memmap so
it fits in 16 GB RAM, and trained with XGBoost on the GPU (8 GB is enough).
Without a CUDA GPU use `--learner lgb --train-frac 0.3` (LightGBM on the CPU,
OOF F0.5 about 0.001 lower).

**Runtime** (20-core laptop CPU, 16 GB RAM, RTX 4060 8 GB): ~2 h 45 min for a
cold run (normalisation ~10 min, train blocking+features ~65 min, stage-1 ~25 min,
stage-2 on the GPU ~45 min, test blocking+features+scoring ~30 min). Re-runs
reuse the caches: changing only model/candidate settings takes ~1 h 30 min.

### Useful flags

| flag | default | meaning |
|---|---|---|
| `--keep-per-source` | 10 | candidates kept per S1 record and source (blended score) before the stage-1 filter |
| `--keep-addr` | 5 | address channel: also keep the top-N per source by address cosine (≥ 0.3) |
| `--stage1-recall` | 0.985 | share of blocked true pairs the stage-1 filter must keep |
| `--resolve-conflicts/--no-resolve-conflicts` | on | one-owner post-processing of the final matches |
| `--k-key` | 30 | neighbours retrieved per S1 record and source |
| `--key-max-df` | 0.0005 | drop blocking keys shared by more than this share of a bucket |
| `--feature-set` | v3 | `v1` = original features; `v2` = + house-number / name-rarity / script; `v3` = + name uniqueness and legal-form relation |
| `--key-feature/--no-key-feature` | on | retrieval score as a feature |
| `--learner` | xgb | stage-2 learner: `xgb` (XGBoost hist, GPU) or `lgb` (LightGBM, CPU) |
| `--lr`, `--num-leaves`, `--max-rounds` | 0.1, 127, 3000 | stage-2 learning rate / leaves (XGBoost `max_leaves`, lossguide) / round cap |
| `--train-frac` | 1.0 | share of train S1 used for stage-2 fitting / OOF |
| `--stage1-frac` | 0.3 | share of train S1 used for stage-1 fitting / OOF |
| `--emb-feature` | off | multilingual name-embedding cosine (tried, no gain; needs `embed_feature.py` + sentence-transformers) |
| `--prune-score`, `--num-variants` | cheap, off | alternatives tried and rejected (see `artifacts/experiments.md`) |
| `--cross-country` | off | ignore the country label in blocking |
| `--skip-test` | off | train + validate only |
| `--rebuild` | off | ignore cached blocking/feature chunks |

Deterministic: fixed seeds, deterministic S1 sample.

## Licences

The trained models are XGBoost (Apache-2.0; stage-2 matcher) and LightGBM (MIT;
stage-1 filter), each a few MB. Libraries: pandas/numpy/scipy/
scikit-learn (BSD), rapidfuzz (MIT), pyarrow and sparse_dot_topn (Apache-2.0),
Unidecode (GPL-2.0+, used only as a text transliteration library, not a model).

## Analysis helpers

```bash
python code/business_entity_resolution/src/error_analysis.py --artifacts-dir artifacts -n 150
python code/business_entity_resolution/src/blocking_recall.py --n-query 40000 --keep-addr 0 3 5
python code/business_entity_resolution/src/tune_candidates.py --artifacts-dir artifacts
python code/business_entity_resolution/src/conflict_eval.py --artifacts-dir artifacts
```

## Package the final submission

```bash
python code/business_entity_resolution/src/make_submission.py --team <team_name>
```
