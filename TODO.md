# TODO — picked up from the 2026-07-28 session

Remaining work. Project context is in **`CLAUDE.md`**; section references (§) point there.

Split into **pipeline** work (code) and **Sohan** work (data, decisions, professor follow-ups).

---

## 0. State as of 2026-07-28

**The recalculation has been run.** It exposed an int32 overflow in pixy's `count_comparisons`
(CLAUDE.md §6.5), which has been fixed and the recalculation re-run.

| file | change |
|---|---|
| `ToUseOnBeagles/CallableSites.py` | **new** (07-27) — per-chromosome callable-site denominators (provisional: pixy `window_pos_2`) |
| `ToUseOnBeagles/AverageData.py` | rewritten (07-27) — pools across chromosomes, converts π/d_xy to per-site; **07-28: denominators now analytic, with a representability cross-check that raises** |
| `ToUseOnBeagles/CalcGenRel.py` | callable-site denominator; **row order fixed** to the specifier matrix; headerless output |
| `Python_Code/ABCAnalysisNoRedis.py` | `abc_results.csv` header written when the file is empty, not just absent |

Verified 07-28: a synthetic round-trip fixture recovers known π/d_xy to ~1e-9 under both int32
saturation and modular wrap, and raises on a non-overflow mismatch; the saturating-row count
predicted from sample sizes alone is **244, matching the real run exactly**; the full three-year
run completes without raising, which positively confirms `count_missing = 0` and that specifier
sample counts match pixy's popfile.

**`data/empiricalStats/*.csv` still holds the OLD per-SNP targets.** The corrected output is in
`data/empiricalStats_new/`, **not yet validated and not yet promoted.** Nothing downstream has
changed.

> The copy of `data/empiricalStats_new/` pulled *before* the overflow fix is contaminated: 2015
> π/d_xy only, 244 rows, of which only 18 are visibly wrong. If there is any doubt about which
> copy is on disk, re-pull rather than inspect — `H53-2015` π is negative in the bad one.

### Immediate next steps

- [ ] **Validate the re-run `empiricalStats_new`** before promoting it. What to check:
      2015's old/new π ratio uniform at ~11.5 across all 24 rows (it was 13.19, with H53 at
      −5.67); `H53-2015` π ≈ 0.0124, not negative; `Alsum25 × Refuge` and `Alsum25 × H15` d_xy
      down from ~0.0148 to ~0.0126; no negative π or d_xy anywhere; **2019/2023 bit-identical to
      the earlier pull** (zero rows changed for those years, so any difference is a red flag).
      Mean per-site π printed 2015 → 0.0121171, and 2019/2023 were 0.0123/0.0122.
- [ ] **Promote:** rename `empiricalStats` → `empiricalStats_old`, `empiricalStats_new` →
      `empiricalStats`, so the code paths stay unchanged.
- [ ] **Commit and push** the promoted targets *and* `ToUseOnBeagles/AverageData.py` together —
      CHTC clones from `origin/main`, so unpushed targets mean CHTC scores against the old
      per-SNP values while local runs score against the new ones. The two result sets would look
      poolable and would not be. Committing the script alongside keeps the provenance of the
      numbers with the numbers.
- [ ] **Delete `Python_Code/abc_results.csv`** (header-only, pre-refactor column layout).
      Precautionary — it has zero data rows.
- [ ] **Never pool ABC results produced before this fix** with results produced after. Rejection
      ABC pools trivially across jobs and submissions, which is exactly what makes it easy to
      concatenate incompatible batches by accident.
- [ ] **Optional:** add the overflow fixture test to the repo. `ToUseOnBeagles/` has no tests, and
      this bug class is invisible in the output for any subpop under 14 individuals. The working
      version is in the session scratchpad (`test_averagedata.py`); it needs the `SRC` path made
      relative before it is worth committing.

---

## 1. Blocking the big pass

- [ ] **Resolve `ancestral_Ne = 6700`** (§6.1). The denominator fix removes ~10× of the ~2400× μ
      inflation; a **~217× mismatch remains**. Needs the provenance question answered (§3 below).
      Then reset it to a defensible long-term coalescent value.
- [ ] **Rebuild the μ prior** around the biological rate (~2.1e-9). The current
      `lognorm(s=0.5, scale=5e-6)` is calibrated to the broken π target and is meaningless after
      the fix. `Main.main`'s default μ needs the same treatment.
- [ ] **Re-derive whether N is identifiable** (§6.2) under corrected values — **do not assume the
      old conclusion holds.** If ancestral Ne rises ~217× and μ drops to biological, the
      forward/ancestral balance shifts. Use branch-mode diversity to diagnose; `diagnostics/qdriver.py`
      is the harness. Keep POPMULT ≤ ~20000 per arm, or fix §6.4 first.
- [ ] **Re-run the integration test** and confirm `pi_loss` is informative — it was a saturated
      ~7.8 at biological μ, and 0.21 under the broken scale. Do this before spending CHTC compute.

---

## 2. Pipeline / code

- [ ] **`recombination_rate` never reaches SLiM** (§6.3). `CPBSampleSimLinux.slim:12` hardcodes
      `initializeRecombinationRate(1e-8)`; the ABC value goes only to recapitation. Pass it as a
      `-d` constant like `POPMULT`. Prerequisite for any LD work.
- [ ] **Remove `simplificationRatio=INF`** (§6.4). `CPBSampleSimLinux.slim:4` never simplifies
      during the forward run — the likely real cause of the historical OOM crashes, not μ. Letting
      SLiM auto-simplify is untested and is the obvious first thing to try. If it works, large
      POPMULT becomes affordable and the "N crashes the machine" constraint dissolves.
- [ ] **Pool F_st properly** (§5.2) — optional. pixy emits no variance components, so
      `AverageData.py` uses a SNP-weighted mean. Proper pooling needs `allel.weir_cockerham_fst`
      recomputed from the VCFs. Error is a few percent and F_st is scale-invariant, so this is the
      lowest-stakes item here.
- [ ] **Tajima's D as a posterior-predictive check** (not fitted). Attractive because it comes
      from the SFS and is **immune to the denominator problem entirely**. Sim: `ts.Tajimas_D()`.
      Empirical: per-subpop SFS via `allel.tajima_d` on the VCFs already loaded in `CalculateLD,py`.
      **Never MAF-filter first** — rare variants are the whole signal. Caveat: imputation bias hits
      rare variants hardest, biasing D upward.
- [ ] **LD statistic for recombination** — deferred, low priority. Blockers: recombination must
      reach SLiM first; 1e6 bp is too short for ~100 kb LD decay (few independent long-range
      pairs); Beagle imputation inflates empirical LD vs tskit's perfect phase (systematic, biases
      r downward); ρ=4N_er is another compound parameter (another ridge). Use tskit
      `ld_matrix(stat="r2")` directly, not via VCF (`write_vcf` rounds positions → duplicates on
      short sequences). Match MAF/bins/MAX_DIST/biallelic to the empirical side.
- [ ] **Optional cleanup:** add `##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">` to
      `ConvertBeagleToVCF.py::generate_VCF_header` to silence the scikit-allel warning (§5.4).
      Cosmetic; only matters if VCFs are ever regenerated.

---

## 3. Sohan — data, decisions, provenance

- [ ] **Ask the professor about `ancestral_Ne = 6700`.** The single highest-value unknown:

  > The 6700 figure we've been using for ancestral Ne — what was it estimated from, and is it a
  > contemporary/local estimate or a long-term coalescent one? Recapitation needs the long-term
  > coalescent Ne, and our corrected empirical π (~0.012 per site) implies something around 1.5
  > million at the biological mutation rate. If 6700 is a contemporary estimate, I think we've
  > been using it in the wrong role — and that's what's been forcing our mutation rate ~200×
  > above the biological value.

- [ ] **Set the `total_migration` prior ceiling** deliberately — currently the placeholder
      `U(0.001, 0.301)` at `ABCAnalysisNoRedis.py:27`. Needs a biological CPB immigration ceiling
      from the dispersal literature or the professor.
- [ ] **Decide on the duplicated `Arlington` label** (§4) — two near-identical subpopulations from
      a typo. Merge, or keep deliberately.
- [ ] **Check the empirical IBD slope is significantly ≠ 0** before trusting it. CPB is a recent,
      fast-spreading Wisconsin invader, so Rousset's drift–dispersal equilibrium assumption is
      questionable (Whitlock & McCauley 1999). The measured slopes are weak and mixed in sign
      (2015 −1.42e-3, 2019 +8.04e-4, 2023 +1.44e-4). **If the observed slope isn't significant,
      `scale` is effectively unconstrained and its posterior is meaningless** — worth knowing
      before the pass, since IBD is one of only three fitted statistics.
- [ ] **Optional:** get reference assembly chromosome lengths. Would give the other bound on
      callable sites (§5.1) and confirm that `chr{i}_cpb.vcf.gz` really is assembly chromosome
      `i` — currently unverifiable, since `ConvertBeagleToVCF.py` hardcodes `CHROM = 9` (§5.4).
      Not blocking; the bracket is ~10–20% against a ~2400× correction.

---

## 4. Running the pass (once §1 clears)

- [ ] **Measure the noise floor FIRST.** Run 2–3 replicates at identical parameters and compute
      the distance `D` between them. If the acceptance threshold ε lands below that floor, the ABC
      is selecting on coalescent/subsampling noise rather than signal.
- [ ] **Smoke test:** 5 jobs × 5 trials. Confirm each `abc_results.csv` has exactly 5 rows **and a
      header row** (the §6.5 fix), and that `slim` is on PATH and `cpb-env` activates inside the
      container. Clear any empty result CSVs from earlier failed runs so they aren't concatenated.
- [ ] **First real batch:** many short jobs (e.g. 250 × 20 trials, not 100 × 50). At ~4 min/trial,
      ~2 h/job survives OSPool eviction far better than ~5 h/job, and rejection ABC pools trivially
      across jobs. **Bump the `OFFSET`/seed per batch so job ids never repeat** — repeated ids
      re-draw identical parameter sets and bias the posterior.
- [ ] **After the pass:** run `abc_standardize.py` (offline σ=1.4826·MAD → standardized `D` →
      ranked CSV), then inspect posteriors. **Report θ=4Nμ and Nm**, not a standalone N peak, and
      check whether the N marginal escapes the prior. Use relatedness and d_xy (and Tajima's D, if
      added) as posterior-predictive checks — statistics you didn't fit are the real validation.
      Remember `numClusters` in the CSV is the raw draw; multiply by 33.
- [ ] **Manual ABC-SMC iteration**, if wanted: take the top ~20% of trials, resample the prior
      around them, run another batch, repeat. Effectively SMC by hand. Note a single big rejection
      batch is unbiased with no importance weights; manual iteration needs them to stay that way.
