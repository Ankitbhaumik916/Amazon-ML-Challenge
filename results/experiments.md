# Experiment log

Metric: out-of-fold macro F0.5 on train (GroupKFold(5) grouped by Source-1 id),
all evaluated Source-1 entities included (singletons count 1.0 when predicted
empty). Model fitting / OOF uses a deterministic 30 % sample of train S1
entities (`--train-frac 0.3`, ~662k entities, ~2.9M candidate pairs); blocking
recall, candidates/entity and the blocking ceiling are measured on all 2.2M
train S1. A change is kept only if OOF F0.5 improves.

"cands/S1" = average size of the final candidate list (= `candidate_pairs.tsv`,
after the stage-1 filter).

| # | change | blocking recall (pre-stage-1) | cands/S1 | stage-1 recall | ceiling F0.5 | OOF F0.5 | India | US | kept? |
|---|---|---|---|---|---|---|---|---|---|
| R0 | **Baseline** (original features/params; scalable key-retrieval blocking k=30, prune by `cheap` to 6/source, stage-1 recall 0.995, LightGBM lr 0.05 / 63 leaves / ≤3000 rounds) | 0.9150 (11.62/S1) | 4.34 | 0.9105 | 0.9679 | **0.9383** | 0.9180 | 0.9519 | – |
| R1 | + normalize v2: transliterated legal words by skeleton (praaivett→pvt, limittedd→ltd, प्रा. लि.), leetspeak (hea1thcare), dotted acronyms (L.L.C.→llc), more US/India/French legal forms (PC, EI, SELARL, SCP…), French street types/articles (R/Rue, Bd, Av, de/la/du), `null`/`n/a` placeholders, leading zeros in house numbers, web domains / store-ID numbers in names | 0.9218 (11.62/S1) | 4.38 | 0.9172 | 0.9707 | **0.9440** | 0.9276 | 0.9550 | ✅ (+0.0057) |
| R2 | + feature set v2: house-number relations (Levenshtein of first numbers, one-digit-drop flag, numeric-core equality, log abs difference), space-insensitive name ratio, skeleton-token Jaccard, mean IDF of each name (coined brand names), non-Latin-script flag | 0.9218 | 4.38 | 0.9172 | 0.9707 | **0.9542** | 0.9363 | 0.9662 | ✅ (+0.0102) |
| R3 | + blocking keys robust to a dropped first/last house-number digit (`--num-variants`) | 0.9212 | 4.37 | 0.9166 | 0.9704 | 0.9538 | 0.9361 | 0.9657 | ❌ (−0.0004) — retrieval was not the bottleneck; the `cheap`-score pruning is |
| R4 | + `--keep-per-source 10` (wider list; the learned stage-1 filter does the pruning instead of the hand-made `cheap` score) | 0.9399 (18.28/S1) | 4.79 | 0.9352 | 0.9774 | **0.9591** | 0.9449 | 0.9686 | ✅ (+0.0049) |
| R5 | + stage-2 LightGBM capacity: learning rate 0.1, 127 leaves (R0–R4 hit the 3000-round cap in every fold; now early-stops at ~1400–1800) | 0.9399 | 4.79 | 0.9352 | 0.9774 | **0.9594** | 0.9455 | 0.9687 | ✅ (+0.0003) |
| R6 | + retrieval score `cos_key` as a stage-1 and stage-2 feature (`--key-feature`) | 0.9399 | 4.63 | 0.9352 | 0.9773 | **0.9596** | 0.9455 | 0.9690 | ✅ (+0.0002, and −0.16 cands/S1) |
| R7 | + `--stage1-recall 0.985` (step 6c: smaller candidate set; pre-screened on R4 OOF, see `runs/R4_candidate_tuning.tsv`) | 0.9399 | **4.18** | 0.9260 | 0.9743 | 0.9584 | 0.9437 | 0.9682 | ✅ by the 6c rule (−0.0012 < 0.002 for −0.45 cands/S1) |
| R8 | `--stage1-recall 0.98` (next step down) | 0.9399 | 4.08 | 0.9213 | 0.9728 | 0.9571 | 0.9422 | 0.9671 | ❌ (−0.0025 vs best 0.9596 exceeds the 0.002 budget) → **R7 setting (0.985) is final** |

### Round 2 (blocking-recall focus, after error analysis of R7 — see `artifacts/error_analysis_R7.txt`)

Blocking recall = pair recall of the pruned list before the stage-1 filter, all 2.2M train S1.

| # | change | blocking recall | cands/S1 | stage-1 recall | ceiling F0.5 | OOF F0.5 | India | US | kept? |
|---|---|---|---|---|---|---|---|---|---|
| R9 | + address channel in pruning: also keep the top-5 candidates per source by address cosine (≥0.3) — recovers coined/brand, web and transliterated names at the same address that the name-heavy `cheap` score pruned. Recall check (80k queries): 0.9360 → 0.9550 | 0.9574 (22.72/S1) | 4.27 | 0.9432 | 0.9805 | **0.9645** | 0.9524 | 0.9726 | ✅ (+0.0061) |
| – | fix 2a: transliteration-skeleton keys (skeleton of un-abbreviated tokens + soft-g / final-j rules, e.g. services ~ sarvisej) and glued-name keys (`rosaleennevarezplum`) added to the key vector — blocking-recall check only (80k queries, with address channel): 0.9550 → **0.9472** (extra name keys dilute the address keys) | – | – | – | – | not run | – | – | ❌ failed recall check |
| – | fix 2b: the same keys as a separate name-only top-10 retrieval channel — retrieval@30 0.9605 → 0.9638, but pruned-list recall 0.9550 → **0.9503** (look-alike names push true pairs out of the `cheap` top-10; India −1.0 pt) | – | – | – | – | not run | – | – | ❌ failed recall check |
| R10 | + feature set v3: name uniqueness (how many S1 / target records of the bucket share each side's exact core name) and legal-form relation (equal / extra on either side / conflicting legal suffixes) — targets empty-address false negatives and "sibling" false positives | 0.9574 | 4.27 | 0.9432 | 0.9805 | **0.9691** | 0.9576 | 0.9767 | ✅ (+0.0046) |
| – | fix 4: state/region canonicalisation (TX↔Texas, RJ↔Rajasthan↔raajsthaan, Telugu/Devanagari/Malayalam state names, French departments → region) — blocking-recall check (80k queries): 0.9550 → **0.9551** (flat; pairs/S1 22.73 → 22.57) | – | – | – | – | not run | – | – | ❌ recall did not improve → not run, reverted |
| R11 | step-4 re-tune (recall now > 0.95): pre-screen on R10 OOF (`runs/R10_candidate_tuning.tsv`) → `--keep-per-source 8 --keep-addr 5 --stage1-recall 0.98`, confirmed by this run | 0.9551 (19.92/S1) | **4.13** | 0.9362 | 0.9782 | 0.9672 | 0.9554 | 0.9751 | ✅ by the 6c rule (−0.0019 < 0.002 for −0.14 cands/S1) — **final** |
| FINAL | full pipeline incl. test with the R11 defaults (`artifacts/runs/FINAL2.log`): reproduces R11 exactly; test 19.60 blocked → **4.89 candidates/S1**, predicted singletons 6.3 % (India 6.5 %, US 5.9 %, France 6.8 %); validator PASS | 0.9551 | 4.13 | 0.9362 | 0.9782 | **0.9672** | 0.9554 | 0.9751 | – |

Step-5 sanity check (R7 OOF): 930,850 candidate pairs with token-set > 95 on both name and address; 90.9 % are true matches, the model predicts 90.8 % as matches; the 30 random samples are all scored correctly (p = 1.00 for true matches, ≈ 0 for non-matches) and every low-scored near-identical pair inspected is a genuine non-match ("sibling" business: extra word / different legal type / neighbouring house number). No scoring bug.

### Round 3 — one-owner conflict resolution and final configuration

| # | change | blocking recall | cands/S1 | stage-1 recall | ceiling F0.5 | OOF F0.5 | India | US | kept? |
|---|---|---|---|---|---|---|---|---|---|
| R12 | + `--resolve-conflicts` (post-processing): an S2/S3 record predicted for several S1 entities is kept only for the highest-probability one (ties → higher name+address cosine). Evaluated with `conflict_eval.py` on R11: conflicts involve 0.033 % of predicted matches; OOF 0.96721 → 0.96727 (+0.00006); the winner is the true owner in 57 % of conflicts (clear wins for sibling businesses at one address, coin flips for empty-address / coined-name records) | 0.9551 | 4.13 | 0.9362 | 0.9782 | 0.9673 | – | – | ✅ (tiny, consistent at every threshold) |
| **FINAL** | **R10 settings (10 per source + 5 address channel, stage-1 recall 98.5 %, feature set v3) + conflict resolution** — chosen over R11 because +0.0019 OOF on the primary metric outweighs 0.14 fewer candidates/S1 (a secondary tiebreaker). `runs/FINAL3.log`: OOF 0.9690 without / **0.9692 with** conflict resolution (`runs/conflict_eval_final.txt`: 0.96905 → 0.96915); test 22.38 blocked → **5.09 candidates/S1**, predicted singletons 6.2 % (India 6.4 %, US 5.8 %, France 6.8 %), 1,319 test records resolved (1,475 pairs dropped, 33 ties); validator PASS | 0.9574 (22.72/S1) | **4.27** | 0.9432 | **0.9805** | **0.9692** | 0.9576 | 0.9767 | – |

Note: the final run gives 0.9690 vs 0.9691 in R10 for the same settings — LightGBM multi-threaded training is not bit-for-bit deterministic across runs.

### Round 4 — deadline sprint (27 Sep; backup of the 0.953-leaderboard package in `backup_final_0953/`)

| # | change | OOF F0.5 | kept? |
|---|---|---|---|
| S1 | expected-F0.5 per-S1 decision rule (top-k maximising expected F0.5, Poisson-binomial exact; raw or nested-isotonic-calibrated probabilities; missed-match prior 0/0.05/0.15) vs tuned threshold rule (0.96905) — `decision_eval.txt`. Probabilities already well calibrated (slightly over-confident in 0.4–0.8). Best: 0.96828 (isotonic, prior 0) | 0.96828 | ❌ (−0.00076: candidates of one S1 are strongly correlated duplicates, the independence assumption hurts) |
| S2 | France diagnostics on test (`test_diag_before.txt`): uncertain-zone share (top p 0.3–0.8) France 0.93 % / India 0.84 % / US 0.59 %; top p < 0.3: 5.9 / 5.7 / 5.3 %; predicted empty 6.75 / 6.40 / 5.82 %. 40 uncertain French S1 read: nearly all are "sibling" businesses (extra word Groupe/Holding/Participations/Développement, other legal form, neighbouring number) or empty candidate addresses — French abbreviations (R./Rue, Av., Crs, Nº, S.A.S.U., Frères, Ets, St-/Saint-, curly apostrophes, bis) are already normalised; only `Q.` (quai) and `Fg` were missed. 20 "nothing close" French S1 are generic names without any near-duplicate (genuine singletons) | – | ❌ no French-specific gap worth a re-normalise + re-block (~2.5 h) |
| S3 | stage-2 on **all** train S1 (`--train-frac 1.0`, 9.4M pairs; stage-1 unchanged, fitted on the same 30 % sample) with **XGBoost hist on the RTX 4060** (`--learner xgb`, lossguide 127 leaves, eta 0.1, same GroupKFold(5)/early-stopping protocol); stage-2 matrix streamed to a float32 memmap. Comparable score on the same 30 % S1 subset: 0.96905 → **0.97034** (+0.0013); with conflict resolution 0.96915 → **0.97037**; all 2.2M S1: 0.9704 (India 0.9593, US 0.9777). Every fold hit the 3000-round cap | **0.97034** (subset) | ✅ |
| S4 | step 4: multilingual name-embedding cosine (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, Apache-2.0, 118M params, fp16 on the GPU; original names, only pairs whose S2/S3 name is non-Latin script — 7–16 % of India pairs; mean cosine true 0.415 vs false 0.297) **+** `--max-rounds 5000` (folds stopped at 3064–3475), on top of S3 | 0.97034 (subset) / 0.97038 resolved | ❌ no gain (+0.00001 = noise) → abandoned, embedding and its dependencies not shipped |
| **FINAL** | S3 configuration (XGBoost GPU stage-2 on all train S1, 3000-round cap, R10 blocking/features, conflict resolution) — `runs/FINAL4.log`: OOF **0.9704** on all 2.2M train S1 (**0.97039** with conflict resolution; India 0.9593, US 0.9777); comparable 30 % subset 0.97034 / 0.97037 (previous final 0.96905 / 0.96915, +0.0012); blocking ceiling 0.9805; train 4.27 cands/S1; test 22.38 blocked → **5.09 candidates/S1**, predicted empty 6.3 % (India 6.4 %, US 5.8 %, France 7.0 %), 629 test records resolved (676 pairs dropped, 29 ties), 3.64 % of test S1 match lists changed vs the 0.953 package; validator PASS | **0.97039** | final |

France diagnostics on test, before (0.953 package) → after (final): uncertain-zone share 0.93 % → 0.88 %; top probability < 0.3: 5.92 % → 6.19 %; predicted empty 6.75 % → 6.96 % (India 6.40 → 6.42 %, US 5.82 → 5.83 %). Blocked / candidate counts unchanged (23.8 / 4.97 per S1). (`test_diag_before.txt`, `test_diag_after.txt`)
