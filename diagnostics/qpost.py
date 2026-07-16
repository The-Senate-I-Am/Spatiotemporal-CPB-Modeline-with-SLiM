"""Post-process an existing SLiM .trees file: recapitate -> simplify -> mutate -> stats.

Reports BRANCH-mode diversity (= 2*E[T_pair] in generations, independent of mu) alongside
site-mode pi. Since mutations are overlaid post-hoc, site_pi = mu * branch_div exactly, so
branch_div isolates what the DEMOGRAPHY is doing with no mutation-rate confound.
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import tskit, msprime, pyslim


def genome_indices(cluster_data, year):
    a = cluster_data[f"Genome Assignment {year}"]
    idx = [-1] * (int(np.nanmax(a.dropna())) + 1)
    for i in range(len(a)):
        if not pd.isna(a[i]):
            k = int(a[i])
            if idx[k] == -1:
                idx[k] = i
    return idx


p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--outdir", default="../out")
p.add_argument("--mu", type=float, default=2.1e-9)
p.add_argument("--recap-rho", type=float, default=2.75e-6)
p.add_argument("--anc-ne", type=float, default=6700)
p.add_argument("--times", default="16,8,0")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--label", default=None)
a = p.parse_args()

outdir = Path(a.outdir)
times = dict(zip(["2015", "2019", "2023"], [int(x) for x in a.times.split(",")]))
label = a.label or a.tag

ts = tskit.load(str(outdir / f"trees_{a.tag}.trees"))
ts = pyslim.recapitate(ts, recombination_rate=a.recap_rho, ancestral_Ne=a.anc_ne,
                       random_seed=a.seed)

cd = pd.read_csv(outdir / f"cluster_data_{a.tag}.csv")
gi = {y: genome_indices(cd, y) for y in ["2015", "2019", "2023"]}

keep = []
for y in ["2015", "2019", "2023"]:
    for i in gi[y]:
        keep.extend(ts.samples(population=i, time=times[y]))
ts = ts.simplify(samples=keep)

ts_mut = msprime.sim_mutations(
    ts, rate=a.mu,
    model=msprime.SLiMMutationModel(type=0, next_id=pyslim.next_slim_mutation_id(ts)),
    keep=True, random_seed=a.seed)

# migration probabilities SLiM actually applied
R = np.loadtxt(outdir / f"migration_rates_{a.tag}.csv", delimiter=",")
sub = R[1:, 1:]
immig = np.array([sub[i].sum() - sub[i, i] for i in range(sub.shape[0])])

res = {"label": label, "tag": a.tag, "mu": a.mu, "anc_ne": a.anc_ne,
       "mean_immigration_applied": float(immig.mean())}

for y in ["2015", "2019", "2023"]:
    ss = [ts.samples(population=i, time=times[y]) for i in gi[y]]
    branch = np.array([ts.diversity([s], mode="branch")[0] for s in ss])
    site = np.array([ts_mut.diversity([s], mode="site")[0] for s in ss])
    k = len(ss)
    F = []
    for i in range(k):
        for j in range(i + 1, k):
            F.append(float(np.asarray(ts_mut.Fst([ss[i], ss[j]])).ravel()[0]))
    res[y] = {
        "n_subpops": k,
        "branch_div_mean": float(branch.mean()),  # 2*E[T_pair], generations; = pi/mu
        "site_pi_mean": float(site.mean()),
        "site_pi_sd": float(site.std()),
        "fst_mean": float(np.nanmean(F)),
    }

(outdir / f"result_{label}.json").write_text(json.dumps(res, indent=2))
print("DONE " + json.dumps(res))
