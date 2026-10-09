# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** TangoAlpha  
**Team Members:**
- Ankit Bhaumik
- Samarth Agarwal
- Bhavneet Dhawan
- Shivam Rajhuvanshi  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
A supervised pairwise-classification pipeline over a small, scalable candidate
set. Records are normalised (transliteration, canonical abbreviations,
script-independent phonetic skeletons, legal-suffix removal), blocked with an
IDF-weighted sparse top-k search over discrete name/address keys, filtered by a
learned stage-1 model to **5.09 candidates per Source-1 entity
on test** (4.27 on train), scored by an XGBoost matcher (trained on the GPU on all
2.2M training entities) whose decision
threshold is tuned directly for macro F0.5, and post-processed so that each
Source-2/3 record is assigned to at most one Source-1 entity. Out-of-fold macro
F0.5 on train is **0.9704** (India 0.9593, US 0.9777) against a blocking ceiling
of 0.9805.

---

## 2. Methodology

### 2.1 Problem Analysis
EDA on the full data (train: 2.21M S1 / 5.03M S2 / 5.29M S3 records; test:
1.73M / 4.89M / 5.08M):

- **Countries:** train has US (60 %) and India (40 %) in every source; test adds
  France (15 % of test S1). Labels are consistent across sources (`US`, `India`,
  `France`), none are missing, and every true train pair has the same country on
  both sides — so the country is a safe blocking bucket, handled as an open set.
- **Match structure:** 5.6 % of S1 entities are singletons; on average an S1
  entity has 3.46 matches (1.7 in S2, 1.8 in S3), i.e. S2/S3 contain several
  duplicates of the same business. 73–75 % of S2/S3 records match some S1.
- **Scripts:** S1 names are Latin script, but **15 % of S2 and 11 % of S3 names
  are written in Indic scripts** (Devanagari, Gujarati, Tamil, Kannada,
  Malayalam, Odia, Bengali). Transliterated with Unidecode they become e.g.
  `raam maarketting praaivett limittedd` ("Ram Marketing Private Limited"), so
  legal words were not recognised and the core names never aligned.
- **Name noise:** legal-suffix insertion/removal/reordering (`Pvt Ltd`,
  `LLC`, `(Corp.)`, `L.L.C.`, `SARL`, `EURL`), word transpositions, typos and
  leetspeak (`Hea1thcare`, `5ervices`, `c0rp`), web domains as names
  (`ridgefresenius.com`), store ids (`#36918`, `(ID: 78239)`), coined brand /
  DBA names that share nothing with the legal name (`Brixorbisol`, `Onyxecto`).
- **Address noise:** abbreviations (Rd/Road, BD/Boulevard, R./Rue), component
  reordering, missing components (3 % of S2/S3 addresses empty; `null`, `N/A`
  placeholders), state names in Indic script, landmark references
  (`Near SBI ATM`), and **house-number noise**: leading zeros (`0227`), letter
  suffixes (`849A`), a dropped digit (`3424`→`424`). Conversely, many *wrong*
  candidates are the same business name on the same street at a nearby number
  (`1197` vs `1201`) — house numbers turned out to be the single most important
  signal.

### 2.2 Solution Strategy

**Approach Type:** Blocking + two-stage classifier (LightGBM candidate filter + XGBoost matcher)  
**Core Innovation:** (1) key-based IDF sparse top-k blocking that scales to
millions of records on a laptop CPU, with a second *address channel* in the
pruning step (0.957 pair recall at 23 candidates/entity); (2) script-independent normalisation (legal words of any
Indic script recognised through their consonant skeleton); (3) house-number
relation features that separate "same entity, noisy number" from "neighbouring
business on the same street", plus name-uniqueness and legal-form-relation
features for empty-address matches and legal-suffix "sibling" businesses; (4) the decision rule is tuned on the exact
leaderboard metric (macro F0.5 including singletons) with grouped out-of-fold
predictions.

Pipeline:
1. **Normalisation** of names and addresses (parallel, cached).
2. **Blocking** — exact top-30 IDF-cosine neighbours over discrete keys in S2
   and in S3, within the same country bucket; exact char n-gram TF-IDF cosines;
   keep the top 10 per source by a blended name/address score plus the top 5 by
   address cosine (address channel).
3. **Stage-1 filter** — LightGBM on vectorised scores; its output *is*
   `candidate_pairs.tsv`.
4. **Stage-2 matcher** — XGBoost (`hist`, GPU) on ~80 name, address, number and context
   features, trained on all training Source-1 entities.
5. **Decision rule** — per S1 entity keep candidates with probability ≥ t
   (and ≥ α × the entity's best probability); t = 0.725, α = 0 tuned for macro
   F0.5 on out-of-fold predictions.
6. **One-owner conflict resolution** — every S2/S3 record belongs to at most one
   S1 entity (true in the training ground truth), so a record predicted for
   several S1 entities is kept only for the highest-probability one (ties:
   higher name+address cosine). Applied to `matching_results.tsv` only; the
   candidate list is not pruned.

Validation: `GroupKFold(5)` grouped by Source-1 id (no entity's candidates leak
across folds). Model fitting and OOF use a deterministic 30 % sample of train
S1 entities (661,897 entities) for the stage-1 filter; the stage-2 matcher uses all
2,206,821 entities (9.4M candidate pairs, streamed to a float32 memmap to fit in
16 GB RAM);
blocking and the stage-1 filter always run on all 2.2M.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** per record a bag of discrete keys — core-name tokens
  (legal forms removed), phonetic-skeleton tokens of the name (catches
  transliterations: Lakshmi/Laxmi, Hindi-script names), adjacent name-token
  bigrams, address words, house numbers, postal codes (5/6-digit) and address
  bigrams. Keys are TF-IDF weighted (sublinear TF, IDF over the country bucket,
  no labels) and keys shared by more than 0.05 % of the bucket (`pvt`, `road`,
  `delhi`) are dropped, so posting lists stay short. Retrieval is the **exact
  top-30 cosine neighbours** per S1 record in S2 and in S3 inside the same
  country bucket, computed with a multi-threaded sparse top-k product
  (`sparse_dot_topn`, Apache-2.0): cost is O(N·K·posting length), not O(N²);
  ~1 minute per million queries. The interface maps directly onto an ANN index
  at larger scale.
- **Scoring and pruning:** every retrieved pair gets exact character n-gram
  TF-IDF cosines of the core name (2–4-grams), the name skeleton (2–3-grams)
  and the address (3–4-grams); hashed features, IDF over the bucket. The list
  is pruned per source to the union of the top 10 by
  `0.65·max(name, skeleton) + 0.35·address` and — the **address channel** — the
  top 5 by address cosine (≥ 0.3). The error analysis showed that 54–67 % of the
  true pairs lost at pruning were coined brand names, web-domain names or
  Indic-script names at the *same address*, which the name-heavy blend ranks
  low; the address channel recovers them (recall check 0.9360 → 0.9550).
- **Stage-1 learned filter:** LightGBM (15 leaves) on the cosines, the key
  score, the blended score, its rank and the gaps to the best candidate in both
  directions (best candidate for the S1 record, best S1 record for the
  candidate), and the address-channel rank. The threshold keeps 98.5 % of the
  blocked true pairs; the
  surviving list is `candidate_pairs.tsv`.
- **Candidate pairs generated:**

| stage (train, all 2.2M S1) | pairs | avg candidates / S1 | pair recall |
|---|---|---|---|
| key retrieval (top-30 per source) | 132.4M | 60 | 0.9605 (80k-query sample) |
| after pruning (top 10 blended + top 5 address) | 50.1M | 22.72 | 0.9574 |
| after stage-1 filter (= `candidate_pairs.tsv`) | 9.4M | **4.27** | **0.9432** |

  Test set: 22.38 per S1 after pruning, **5.09 per S1
  in `candidate_pairs.tsv`** (8,822,685 pairs for 1,732,544 S1).
  Reduction ratio vs. all S1×(S2∪S3) pairs (same-bucket or not): 0.99999780
  before the stage-1 filter and 0.99999959 after it on train; 0.99999949 for
  the submitted test `candidate_pairs.tsv`.
- **How you ensured true matches were not lost:** the blocking ceiling is
  measured on every run (macro F0.5 if the matcher were perfect on the
  candidate set: **0.9805**). Two rounds of error analysis on the missed
  pairs (reading 150 per bucket, and tagging every missed pair by pattern and
  by stage — never retrieved vs. retrieved-then-pruned) drove the design:
  transliteration-robust skeleton keys, address keys that match even when the
  name is a coined brand, IDF weighting so rare keys dominate, a wider pruned
  list cut by a *learned* filter (+0.005 OOF F0.5) and the address channel
  (+0.006). Every blocking change is first checked with a cheap recall-only run
  (`blocking_recall.py`, 80k queries against the full targets) before a full
  run; transliteration/glued-name keys and state canonicalisation were rejected
  there. The candidate widths and the stage-1 recall target were re-tuned after
  recall passed 0.95; the smallest setting within 0.002 F0.5 (8 + 5 per source,
  98 %) saved only 0.14 candidates/entity for −0.0019 F0.5, so the final
  configuration keeps 10 + 5 per source and 98.5 % (F0.5 is the primary metric,
  candidate-set size only a tiebreaker).

---

## 4. Matching Model

**Features used** (all country-agnostic — nothing is hard-coded per country,
so the model transfers to France):
- Name features: char n-gram TF-IDF cosines (core, skeleton); RapidFuzz ratio,
  partial ratio, token-sort and token-set ratios (core and full name);
  Jaro-Winkler; token Jaccard; IDF-weighted token overlap (min and Jaccard);
  skeleton-token Jaccard; space-insensitive ratio (web domains, glued words);
  first-token and exact-core equality; acronym match (IBM ↔ International
  Business Machines); containment; length difference; mean IDF of each name
  (flags coined brand names); non-Latin-script flag; **name uniqueness** —
  how many S1 records and target-source records of the bucket share each side's
  exact core name (a same-name candidate with an empty address is almost surely
  the match if the name is unique, ambiguous if not); **legal-form relation** —
  equal / extra on either side / conflicting legal suffixes (`X Inc` vs `X LLC`,
  the typical "sibling business" false positive).
- Address features: TF-IDF cosine; fuzzy ratio / token-set / partial ratio; word
  Jaccard and IDF-weighted overlap; **house numbers**: number-set Jaccard,
  number conflict, first-number equality, Levenshtein similarity of the first
  numbers, one-digit-drop flag, numeric-core equality (`849` vs `849A`), log
  absolute difference; postal-code agreement (missing → NaN); city/tail
  equality; landmark flag; emptiness and length.
- Other: retrieval key score; address-channel rank; context features — rank and gap to the best
  candidate of the same S1 record and source (for cosine, token-set, address,
  blended score), gap to the best S1 record for the same candidate (reverse
  view), mutual-best flag, number of competing candidates / S1 records, source
  flag, stage-1 probability.

Top features by gain: `a_num_jacc`, `p1` (stage-1 probability),
`a_num_conflict`, `a_num_drop`, `a_num_logdiff`, `a_len_min`, `legal_extra_r`,
`legal_conflict`, `n_idf_mean_r`, `a_postal`.

**Normalisation details:** Unidecode transliteration (é→e, Indic scripts →
Latin); canonical short forms (Corporation→corp, Private→pvt, Limited→ltd,
Street→st, Road→rd, Boulevard/Bd/Bld→blvd, Rue/R.→r, Near→nr, Bengaluru→
bangalore …); transliterated legal words recognised via their consonant
skeleton (`praaivett`/`piraiveett`→pvt, `limittedd`/`limittett`→ltd,
`elelpii`→llp, `प्रा. लि.`→pvt ltd); legal forms removed for the core name
(Inc, LLC, Pvt Ltd, PC, SARL, SAS, EURL, EI, SELARL, SCP …); dotted acronyms
merged (L.L.C.→llc); leetspeak repaired inside words (hea1thcare→healthcare);
web domains and store ids stripped; French articles and `bis` dropped;
`null`/`N/A` placeholders removed; leading zeros stripped from house numbers;
phonetic skeleton (sh/ch→s, ksh/x→ks, w→v, doubled letters and inner vowels
removed).

**Model type:** XGBoost binary classifier (Apache-2.0, a few MB — far below
8B parameters): learning rate 0.1, 127 leaves, feature/bagging fraction 0.8,
`hist` tree method on an RTX 4060 (CUDA), `lossguide` growth with 127 leaves,
early stopping per fold (≈3,000 rounds), final model retrained on all training
pairs with 1.1× the mean best iteration.  
**Threshold selection method:** grid search of (t, α) maximising macro F0.5 —
computed exactly like the leaderboard, singletons included — on the grouped
out-of-fold predictions; selected t = 0.725, α = 0.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9704** out-of-fold with conflict resolution
  (0.97039; 0.9704 without, rounded; GroupKFold(5) by S1 id, all 2,206,821 train S1
  entities); India 0.9593, US 0.9777. Blocking ceiling 0.9805; predicting every
  entity empty would score 0.0558. On test the tuned rule predicts empty lists
  for 6.3 % of entities (train singleton share 5.6 %): India 6.4 %, US 5.8 %,
  France 7.0 %; 3.19 matches per entity on average (train truth: 3.46).

Experiments (each judged only by OOF macro F0.5; full log in
`artifacts/experiments.md`):

| # | change | blocking recall | cands / S1 | ceiling F0.5 | OOF F0.5 | India | US | kept |
|---|---|---|---|---|---|---|---|---|
| R0 | Baseline: original features, 6 per source, stage-1 recall 99.5 %, lr 0.05 / 63 leaves | 0.9150 | 4.34 | 0.9679 | 0.9383 | 0.9180 | 0.9519 | – |
| R1 | + normalisation v2 (transliterated legal words, leetspeak, acronyms, US/India/French forms, placeholders, leading zeros, web/ids) | 0.9218 | 4.38 | 0.9707 | 0.9440 | 0.9276 | 0.9550 | yes |
| R2 | + feature set v2 (house-number relations, name rarity, skeleton Jaccard, no-space ratio, script flag) | 0.9218 | 4.38 | 0.9707 | 0.9542 | 0.9363 | 0.9662 | yes |
| R3 | + digit-drop variants of house-number blocking keys | 0.9212 | 4.37 | 0.9704 | 0.9538 | 0.9361 | 0.9657 | no |
| R4 | + 10 per source, learned stage-1 pruning | 0.9399 | 4.79 | 0.9774 | 0.9591 | 0.9449 | 0.9686 | yes |
| R5 | + LightGBM lr 0.1 / 127 leaves | 0.9399 | 4.79 | 0.9774 | 0.9594 | 0.9455 | 0.9687 | yes |
| R6 | + retrieval score `cos_key` as a feature | 0.9399 | 4.63 | 0.9773 | 0.9596 | 0.9455 | 0.9690 | yes |
| R7 | + stage-1 recall 98.5 % (smaller candidate set) | 0.9399 | 4.18 | 0.9743 | 0.9584 | 0.9437 | 0.9682 | yes (−0.0012 < 0.002 budget) |
| R8 | stage-1 recall 98 % | 0.9399 | 4.08 | 0.9728 | 0.9571 | 0.9422 | 0.9671 | no (−0.0025 vs R6) |
| R9 | + address channel in pruning (top 5 by address cosine) | 0.9574 | 4.27 | 0.9805 | 0.9645 | 0.9524 | 0.9726 | yes |
| – | transliteration-skeleton + glued-name keys (in key vector / as separate channel) | recall check 0.9472 / 0.9503 vs 0.9550 | – | – | not run | – | – | no |
| R10 | + feature set v3 (name uniqueness, legal-form relation) | 0.9574 | 4.27 | 0.9805 | 0.9691 | 0.9576 | 0.9767 | yes |
| – | state/region canonicalisation (TX↔Texas, RJ↔Rajasthan↔raajsthaan, …) | recall check 0.9551 vs 0.9550 | – | – | not run | – | – | no |
| R11 | re-tune after recall > 0.95: 8 per source + 5 address, stage-1 recall 98 % | 0.9551 | 4.13 | 0.9782 | 0.9672 | 0.9554 | 0.9751 | no — −0.0019 for −0.14 cands/S1 |
| R12 | + one-owner conflict resolution (on R11) | 0.9551 | 4.13 | 0.9782 | 0.9673 | – | – | yes (+0.00006) |
| R13 | R10 settings + conflict resolution (LightGBM, 30 % train) — leaderboard 0.953 | 0.9574 | 4.27 | 0.9805 | 0.9692 | 0.9576 | 0.9767 | yes |
| S1 | expected-F0.5 per-S1 decision rule (exact Poisson-binomial; raw or nested-isotonic probabilities) | – | – | – | 0.9683 | – | – | no (−0.0008) |
| S3 | stage-2 XGBoost on the GPU, trained on all train S1 | 0.9574 | 4.27 | 0.9805 | 0.9704 | 0.9593 | 0.9777 | yes (+0.0012 on the same 30 % subset) |
| S4 | + multilingual name-embedding cosine (MiniLM, Apache-2.0) + 5000-round cap | 0.9574 | 4.27 | 0.9805 | 0.9704 | 0.9593 | 0.9777 | no (+0.00001) |
| **Final** | **S3: R10 blocking/features + XGBoost (GPU, all train S1) + conflict resolution** | 0.9574 | **4.27** | **0.9805** | **0.9704** | 0.9593 | 0.9777 | **final** |

Blocking recall = true-pair recall of the pruned list before the stage-1
filter, on all 2.2M train S1; cands / S1 = size of the final candidate list
(`candidate_pairs.tsv`). The final settings are the pipeline defaults.

Error analysis of R7 on 100k train S1 (346,480 true pairs; every pair tagged,
150 per bucket read — `artifacts/error_analysis_R7.txt`): 6.05 % of true pairs
missed by blocking (≈4.0 pts never retrieved, ≈2.4 pts retrieved then pruned),
1.40 % dropped by the stage-1 filter, 2.05 % false negatives of the matcher,
2,718 false positives.

- **Common false positives (wrong merges):** 54 % are "sibling" businesses — the
  same core name with a different or added legal suffix (`Baptist Association`
  vs `Baptist Association Corp`), usually at a *nearby* house number (4100 vs
  4102); others: same address with an unrelated or coined name, a
  transliterated name of a *different* business at the same address
  (`Royal Media` vs `रॉयल कंसल्टेंसी`), same name with an empty address. The
  legal-form features (R10) target the first group, and the one-owner conflict
  resolution removes the cases where two S1 siblings both claim the same
  record (0.05 % of predicted matches; +0.0001 F0.5). A sanity check of 930k
  near-identical candidate pairs (token-set > 95 on name and address) found no
  scoring bug: all low-scored ones are such siblings.
- **Common false negatives (missed matches):** 38 % have an empty candidate
  address and 36 % an identical name (ambiguous unless the name is unique — the
  name-uniqueness features target this); 15 % a one-digit house-number typo,
  7 % a nearby number. Blocking misses: India — 51 % transliterated names, 47 %
  partial names; US — 68 % partial or typo'd names; retrieved-but-pruned —
  54–67 % coined/brand names at the same address (fixed by the address channel).

---

## 6. Conclusion
A scalable blocking + two-stage gradient-boosting pipeline reaches 0.970 out-of-fold
macro F0.5 with 4.3 candidates per entity on 2.2M training entities (baseline
0.938). The largest gains came from looking at the data: script-independent
normalisation of legal words (+0.006), house-number / name-rarity features
(+0.010), an address channel in pruning that rescues coined, web and
transliterated names (+0.006) and name-uniqueness / legal-form features (+0.005)
mattered far more than model capacity. Every step is
country-agnostic, which lets the same code handle France without training data.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` (all source in `src/`, `README.md`,
`requirements.txt`, `run.sh`/`run.bat`). Entry point, run from
`student_resource/`:

```bash
python code/business_entity_resolution/src/pipeline.py --data-dir dataset --out-dir output --artifacts-dir artifacts
```

It writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` (both
tab-separated) and `artifacts/run_report.json`. Modules: `prep.py` (loading,
parallel normalisation, parquet cache), `normalize.py` (rules), `blocking.py`
(keys, sparse top-k retrieval, exact cosines, pruning, stage-1 features),
`features.py` (vectorised pair and context features), `pipeline.py`
(orchestration, stage-1/stage-2 models, threshold tuning, outputs),
`error_analysis.py`, `blocking_recall.py`, `tune_candidates.py`,
`conflict_eval.py`, `make_submission.py`. Runtime ≈ 2 h 15 min cold on a 20-core laptop CPU with
16 GB RAM (no GPU); every run is deterministic.

### B. Additional Results
- Full experiment log with every change and its OOF score or recall check:
  `artifacts/experiments.md` (rejected: digit-drop blocking-key variants,
  −0.0004; stage-1 recall 98 % at the R7 widths, −0.0025; transliteration /
  glued-name keys and state canonicalisation, failed the recall check).
- Candidate-set trade-off after the address channel (pre-screen on out-of-fold
  predictions, `artifacts/runs/R10_candidate_tuning.tsv`): without the address
  channel F0.5 drops by 0.006–0.016 at every width; with it, 8 + 5 per source
  and a 98 % stage-1 recall is the smallest setting within 0.002 of the best,
  but was not adopted (see §3).
- One-owner conflict resolution (`artifacts/runs/conflict_eval_final.txt`):
  OOF 0.96905 → 0.96915 (LightGBM model); positive at every decision threshold
  tested (0.60–0.80). With the final model trained on all S1 it is evaluated
  directly on the OOF predictions (0.97039 with it; the 30 % subset
  moves 0.97034 → 0.97037).
- Tried and rejected in the last round (`artifacts/experiments.md`, round 4): an
  expected-F0.5 per-entity decision rule (probabilities are well calibrated, but
  the candidates of one entity are strongly correlated duplicates, −0.0008), a
  French normalisation pass (label-free test diagnostics showed no
  France-specific gap: uncertain-zone share 0.93 % vs India 0.84 %) and a
  multilingual MiniLM name-embedding feature for Indic-script pairs (no gain).
  On test it resolves 1,319 records claimed by several S1 entities.
- Per-country OOF F0.5: India 0.9593, US 0.9777. India is harder mainly because
  of the Indic-script names and landmark-style addresses.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
