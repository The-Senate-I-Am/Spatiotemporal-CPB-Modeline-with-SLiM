# TODO — updated 2026-07-29

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

**The corrected targets are promoted and validated.** `data/empiricalStats/` now holds the
per-site values; `data/empiricalStats_old/` holds the old per-SNP ones (untracked). No code
changed — all four consumers (`ABCAnalysisNoRedis.py:226-229`, `ABCAnalysis.py:119,130`) point at
`data/empiricalStats/`, which is exactly why rename-in-place was the right promotion.

Validation passed 07-29, every check from the old list: 2015 mean π **0.0121171** as predicted;
old/new ratio uniform **11.448–11.523** across all 24 rows; `H53-2015` π **0.012371** (not
negative); `Alsum25×Refuge`/`×H15` d_xy **0.012565 / 0.012443**; **zero** negative π or d_xy in
any year; 2019/2023 unchanged; all four matrices match the specifier dimensions 24/17/20.

Two expected-but-unlisted differences, both accounted for: **F_st moved ≤3.7e-3** (the §5.2
SNP-weighted pooling change in `AverageData.py:274-285`, *not* the denominator — F_st is
denominator-invariant), and **relatedness rescaled by a per-year constant** (11.70/11.93/11.80,
identical to 4 s.f. across every entry — the `NORMALISE_BY` switch to callable sites).

### Done 2026-07-29

| item | where |
|---|---|
| `recombination_rate` now reaches SLiM via `-d RECOMB` | `CPBSampleSim{Linux,Win}.slim:12`, `Main.py` (§6.3) |
| `simplificationRatio=INF` removed | `CPBSampleSim{Linux,Win}.slim:4` (§6.4) |
| IBD demoted fitted → diagnostic | `abc_standardize.py::FITTED_STATS`, `calculate_losses` (§7.1) |
| Small-subpop exclusion | `ABCAnalysisNoRedis.py::get_keep_mask` (§7.0) |

### Immediate next steps

- [x] ~~**Run the pipeline end-to-end at low POPMULT.**~~ **Done 07-31.** Runs clean at POPMULT
      500 and 5000; **OOMs at 12000 on a 16 GB machine**, inside recapitation (not SLiM). Full
      timing/memory table and CHTC sizing now in CLAUDE.md §3.1. `-d RECOMB` confirmed threading
      through the real model; recapitation confirmed working on a runtime-simplified tree
      sequence, which is §6.4's safety argument actually exercised.
- [ ] **Commit and push** all of it together — CHTC clones from `origin/main`. Targets, SLiM
      files, `Main.py`, `ABCAnalysisNoRedis.py`, `abc_standardize.py`, `ToUseOnBeagles/AverageData.py`.
- [ ] **Decide whether to commit `data/empiricalStats_old/`.** Currently untracked. Recommend
      **not** committing: the old per-SNP values are already recoverable from git history (the
      previous commit of `data/empiricalStats/`), so committing duplicates history while leaving a
      live wrong-scale directory in the tree for someone to read by accident.
- [ ] **Delete `Python_Code/abc_results.csv`** (header-only, pre-refactor column layout).
      Precautionary — zero data rows. Still present as of 07-29.
- [ ] **Never pool ABC results across the 07-28/07-29 changes.** Rejection ABC pools trivially
      across jobs, which is exactly what makes incompatible batches easy to concatenate by
      accident. The fitted-statistic set itself changed, so old rows are not comparable.
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
      **Empirically pinned 07-31:** at Ne=6700 the value that matches π is **μ ≈ 5.2e-7**
      (measured at POPMULT=5000), against **4.6e-7** from §6.1's independent π arithmetic. Both
      are ~250× biological — so this is not a prior-shape problem, it is §6.1.

- [ ] **BLOCKING — π and F_st currently pull in opposite directions** (§6.2). At μ=5e-6 the
      POPMULT that fits F_st (≈5000) makes π **9.7× too high**; the POPMULT that would help π
      wrecks F_st. **No parameter set satisfies both**, so a pass run now would optimize an
      artifact of the miscalibrated μ rather than anything in the data. This is the single
      clearest reason not to spend CHTC compute until §6.1 is settled.
- [~] **Re-derive whether N is identifiable** (§6.2) — **partially answered 07-31: yes, via F_st.**
      A 500→5000 POPMULT sweep moved simulated F_st from 0.076 onto the observed 0.008, improving
      `fst_loss` **8.7×**, tracking `1/(1+4Nm)`. The old "N is nearly unidentifiable" claim came
      from π alone and does not hold for the structural side. **Still open:** the full sweep with a
      corrected μ, and branch-mode diversity as the diagnostic. Keep POPMULT ≤ ~5000 locally
      (§3.1) or run it where there is more RAM.
- [ ] **Re-run the integration test** and confirm `pi_loss` is informative — it was a saturated
      ~7.8 at biological μ, and 0.21 under the broken scale. Do this before spending CHTC compute.
      Note π is now **one of only two** fitted statistics, so this matters more than it did.

---

## 2. Pipeline / code

- [x] ~~**`recombination_rate` never reaches SLiM**~~ (§6.3) — **done 07-29**, both .slim files.
- [x] ~~**Remove `simplificationRatio=INF`**~~ (§6.4) — **done 07-29**, both .slim files. Safety
      checked against the SLiM/pyslim docs, not assumed. **Not yet run.**
- [ ] **Consider Remembering a bounded subset at gens 308/316** (§6.4). Currently *every*
      individual in *every* subpop is permanently Remembered at both timepoints, pinning all their
      ancestry — a memory floor simplification cannot touch. Only 2–19 per site are ever sampled
      downstream, so ~50/subpop would be ample. Changes what the sampler can draw, so it is a
      design decision. Do this only if memory is still a problem after §6.4.
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

- [ ] **Ask about the collection protocol for `Mortensen9-2015` and `H41-2023`** — the second
      highest-value question after Ne (§7.2):

  > Two of our sites look strongly differentiated from everything else — Mortensen9 in 2015 and
  > H41 in 2023, at roughly 3–6× the next-highest site. But they're also the *lowest*-diversity
  > and *most*-internally-related samples in their year, and they're not geographically remote
  > (the genuinely distant fields aren't differentiated at all). That pattern looks less like a
  > distinct population and more like the beetles in those samples being close relatives — say,
  > collected off a single plant or one egg mass. Do you know how those two fields were sampled,
  > or whether either was newly colonized that season?

- [ ] **Raise the dispersal-kernel specification** in the same conversation (§7.1, §7.2). Two
      independent results say connectivity here is not distance-limited: the IBD slope is null in
      all three years, and F_st structure is site-coherent rather than distance-organized. For a
      pest moved on farm equipment and seed potatoes that is unsurprising, but an exponential
      distance-decay kernel cannot produce "one isolate, rest panmictic." **This is a model
      question, not an ABC-tuning question**, and it may matter more than any prior.
- [ ] **Set the `total_migration` prior ceiling** deliberately — currently the placeholder
      `U(0.001, 0.301)` at `ABCAnalysisNoRedis.py:27`. Needs a biological CPB immigration ceiling
      from the dispersal literature or the professor. Anchor: observed mean F_st ≈ 0.003–0.008
      implies **Nm ≈ 31–83** (§7.1).
- [ ] **Decide on the duplicated `Arlington` label** (§4) — two near-identical subpopulations from
      a typo. Merge, or keep deliberately. **Lower urgency now:** the small-subpop mask already
      drops `Arlington2015` (n=2) from the fitted statistics, so it only affects diagnostics.
- [x] ~~**Check the empirical IBD slope is significantly ≠ 0**~~ — **done 07-29. It is not**, in any
      year (Mantel, 9999 perms: p = 0.639 / 0.148 / 0.985; sign flips across years). IBD demoted
      to diagnostic; **`scale` is unidentifiable and must not be reported as inferred** (§7.1).
- [ ] **Decide whether `scale` should stay a free parameter at all.** With IBD unfitted nothing
      constrains it, so sampling it just adds a nuisance dimension that widens the prior volume
      the pass has to cover. Fixing it would make the pass cheaper for free. Requires picking a
      defensible value — see the kernel-specification item above.
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
- [ ] **Size `request_memory` and `request_disk` from CLAUDE.md §3.1 before submitting.**
      ~8 GB at POPMULT≈5000, **~20 GB at the prior ceiling of 12000**. Undersized memory means
      jobs held or evicted *mid-pass*, and rejection ABC pools whatever survived — silently
      biasing the posterior toward the small-POPMULT draws that fit. Either request for the
      ceiling or lower the ceiling deliberately.
- [ ] **First real batch:** many short jobs. **Re-plan the trial count — it is no longer ~4
      min/trial.** Measured 07-31: **30 min/trial at POPMULT=5000**, projecting ~60 min at 12000
      (§3.1). At those rates 20 trials/job is 10–20 h, far too long for OSPool eviction; think
      **2–5 trials/job**. **Bump the `OFFSET`/seed per batch so job ids never repeat** — repeated
      ids re-draw identical parameter sets and bias the posterior.
- [ ] **After the pass:** run `abc_standardize.py` (offline σ=1.4826·MAD → standardized `D` →
      ranked CSV), then inspect posteriors. **Report θ=4Nμ and Nm**, not a standalone N peak, and
      check whether the N marginal escapes the prior. Use relatedness, d_xy, **the IBD slope**
      (newly available as a check now that it isn't fitted) and Tajima's D if added as
      posterior-predictive checks — statistics you didn't fit are the real validation.
      Remember `numClusters` in the CSV is the raw draw; multiply by 33.
      **Do not report a `scale` posterior** — nothing constrains it (§7.1).
- [ ] **Watch for π/F_st double-counting.** They are strongly correlated in the observed data
      (r = −0.72 in 2015, −0.92 in 2023; §7.2), so equal weights after MAD-standardization partly
      weight one signal twice. Now that the fitted set is only these two, that is the whole
      distance. Consider unequal `WEIGHTS` in `abc_standardize.py` once the pilot batch exists.
- [ ] **Manual ABC-SMC iteration**, if wanted: take the top ~20% of trials, resample the prior
      around them, run another batch, repeat. Effectively SMC by hand. Note a single big rejection
      batch is unbiased with no importance weights; manual iteration needs them to stay that way.
