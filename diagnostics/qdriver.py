"""
Controlled driver for testing coalescent rescaling (Q-invariance) of the CPB pipeline.

Differs from Main.py only in that everything that must scale under rescaling is exposed
as a knob, and the KMeans seed is FIXED so cluster geography is identical across runs
(Main.py randomizes it, which would confound any sweep).

Reports BRANCH-mode diversity as well as site-mode pi. Branch-mode diversity is
E[pairwise coalescence time] * 2, in generations, independent of the mutation rate --
so site pi = mu * branch_div exactly. That strips out mutation Monte Carlo noise and
shows what the demography is actually doing.
"""
import argparse, json, subprocess, random
from pathlib import Path

import numpy as np
import pandas as pd
import tskit, msprime, pyslim

import CollectData, GenerateClusterData, GenerateSimulationParams

SLIM_TEMPLATE = """
initialize()
{{
    initializeTreeSeq(simplificationRatio=INF, timeUnit="generations");
    initializeMutationRate(0);
    initializeMutationType("m1", 0.5, "f", 0.0);
    m1.convertToSubstitution = F;
    initializeGenomicElementType("g1", m1, 1);
    initializeGenomicElement(g1, 0, 1e6 - 1);
    initializeRecombinationRate({slim_rho});
}}

1 early()
{{
    migrationRatesDF = readCSV("{migfile}", colNames=T, sep=",");
    propDF = readCSV("{clusterfile}", colNames=T, sep=",");
    migrationRates = migrationRatesDF.asMatrix();
    numSubpops = propDF.nrow;
    for (i in 0:(propDF.nrow-1)) {{
        sim.addSubpop("p"+i, asInteger(round(propDF.getValue("Average Count")[i] * {popmult} / numSubpops)));
    }}
    for (i in 0:(ncol(migrationRates)-1)) {{
        for (j in 0:(nrow(migrationRates)-1)) {{
            if (i != j)
                sim.subpopulations[i].setMigrationRates(sim.subpopulations[j], migrationRates[i+ncol(migrationRates)*j]);
        }}
    }}
}}

{g1} late() {{ sim.treeSeqRememberIndividuals(sim.subpopulations.individuals, T); }}
{g2} late() {{ sim.treeSeqRememberIndividuals(sim.subpopulations.individuals, T); }}
{g3} late()
{{
    sim.treeSeqOutput("{treefile}");
    sim.simulationFinished();
}}
"""


def genome_indices(cluster_data, year):
    a = cluster_data[f"Genome Assignment {year}"]
    total = int(np.nanmax(a.dropna()))
    idx = [-1] * (total + 1)
    for i in range(len(a)):
        if not pd.isna(a[i]):
            k = int(a[i])
            if idx[k] == -1:
                idx[k] = i
    return idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--popmult", type=float, required=True)
    p.add_argument("--mu", type=float, default=2.1e-9)
    p.add_argument("--recap-rho", type=float, default=2.75e-6)
    p.add_argument("--slim-rho", type=float, default=1e-8)
    p.add_argument("--anc-ne", type=float, default=6700)
    p.add_argument("--gens", type=str, default="308,316,324")
    p.add_argument("--times", type=str, default="16,8,0")
    p.add_argument("--mig-scale", type=float, default=1.0,
                   help="multiply off-diagonal migration PROBABILITIES by this (= Q under rescaling)")
    p.add_argument("--m-modifier", type=float, default=0.0001)
    p.add_argument("--num-clusters", type=int, default=66)
    p.add_argument("--kmeans-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--outdir", type=str, default="../out")
    args = p.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    g1, g2, g3 = [int(x) for x in args.gens.split(",")]
    times = dict(zip(["2015", "2019", "2023"], [int(x) for x in args.times.split(",")]))

    # --- cluster setup with a FIXED seed: identical geography across all runs ---
    random.seed(args.kmeans_seed); np.random.seed(args.kmeans_seed)
    field_data = CollectData.read_csv(Path("../data/final_data_for_modeling.csv"))
    GenerateClusterData.cluster_coordinates(field_data, n_clusters=args.num_clusters,
                                            iters=2000, random_state=args.kmeans_seed)
    clusters = GenerateClusterData.populate_cluster_objects(field_data, estimate_data=True)
    GenerateClusterData.assign_genomes_to_clusters(clusters)

    distfile  = outdir / f"cluster_distances_{args.tag}.csv"
    clustfile = outdir / f"cluster_data_{args.tag}.csv"
    migfile   = outdir / f"migration_rates_{args.tag}.csv"
    treefile  = outdir / f"trees_{args.tag}.trees"
    slimfile  = outdir / f"sim_{args.tag}.slim"

    distances = GenerateClusterData.create_cluster_distance_matrix(clusters, output_path=distfile)
    GenerateClusterData.cluster_data_to_csv(clusters, output_path=clustfile)
    GenerateSimulationParams.determine_migration_rates(
        distances, modifier=args.m_modifier, output_path=migfile)

    # Record the migration probabilities SLiM will actually apply, and optionally
    # rescale them by Q (which is what 4Nm-invariance requires).
    R = np.loadtxt(migfile, delimiter=",")
    sub = R[1:, 1:].copy()          # row0/col0 of the file are raw cluster IDs, not rates
    k = sub.shape[0]
    immig_before = np.array([sub[i].sum() - sub[i, i] for i in range(k)])
    saturated = False
    if args.mig_scale != 1.0:
        for i in range(k):
            off = sub[i].copy(); off[i] = 0.0
            off *= args.mig_scale
            s = off.sum()
            if s >= 0.95:
                saturated = True
                off *= 0.95 / s; s = 0.95
            sub[i] = off
            sub[i, i] = 1.0 - s
        R[1:, 1:] = sub
        np.savetxt(migfile, R, delimiter=",", fmt="%.6f")
    immig_after = np.array([sub[i].sum() - sub[i, i] for i in range(k)])

    slimfile.write_text(SLIM_TEMPLATE.format(
        slim_rho=args.slim_rho, popmult=args.popmult,
        migfile=migfile.as_posix(), clusterfile=clustfile.as_posix(),
        treefile=treefile.as_posix(), g1=g1, g2=g2, g3=g3))

    subprocess.run(["slim", "-l", "0", str(slimfile)], check=True)

    ts = tskit.load(str(treefile))
    ts = pyslim.recapitate(ts, recombination_rate=args.recap_rho,
                           ancestral_Ne=args.anc_ne, random_seed=args.seed)

    cluster_data = pd.read_csv(clustfile)
    gi = {y: genome_indices(cluster_data, y) for y in ["2015", "2019", "2023"]}

    keep = []
    for y in ["2015", "2019", "2023"]:
        for idx in gi[y]:
            keep.extend(ts.samples(population=idx, time=times[y]))
    ts = ts.simplify(samples=keep)

    ts_mut = msprime.sim_mutations(
        ts, rate=args.mu,
        model=msprime.SLiMMutationModel(type=0, next_id=pyslim.next_slim_mutation_id(ts)),
        keep=True, random_seed=args.seed)

    res = {"tag": args.tag, "args": vars(args),
           "mean_immigration_rate_applied": float(immig_after.mean()),
           "mean_immigration_rate_base": float(immig_before.mean()),
           "migration_saturated": bool(saturated)}

    for y in ["2015", "2019", "2023"]:
        ss = [ts.samples(population=i, time=times[y]) for i in gi[y]]
        branch_div = np.array([ts.diversity([s], mode="branch")[0] for s in ss])
        site_pi    = np.array([ts_mut.diversity([s], mode="site")[0] for s in ss])
        kk = len(ss)
        F = np.full((kk, kk), np.nan)
        for i in range(kk):
            for j in range(i + 1, kk):
                F[i, j] = F[j, i] = float(np.asarray(ts_mut.Fst([ss[i], ss[j]])).ravel()[0])
        iu = np.triu_indices(kk, 1)
        res[y] = {
            "n_subpops": kk,
            "branch_div_mean": float(branch_div.mean()),   # = 2*E[T_pair], generations
            "site_pi_mean": float(site_pi.mean()),
            "site_pi_sd": float(site_pi.std()),
            "fst_mean": float(np.nanmean(F[iu])),
        }
    (outdir / f"result_{args.tag}.json").write_text(json.dumps(res, indent=2))
    print("DONE " + json.dumps({k2: v for k2, v in res.items() if k2 != "args"}))


if __name__ == "__main__":
    main()
