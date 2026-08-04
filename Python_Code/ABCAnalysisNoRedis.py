import csv
import math
import numpy as np
import sys
import shutil

from pathlib import Path
from scipy import stats

import Main

# Operating parameter values.
# mutation_rate here is a NUISANCE DIVERSITY-SCALER (~5e-6), NOT the biological rate (~2.1e-9):
# with ancestral_Ne fixed at 6700, pi = 4*Ne*mu, so matching empirical pi (~0.14) needs mu ~5e-6.
# mu is not interpreted biologically (CLAUDE.md 5.1).
DEFAULT_MUTATION_RATE = 5e-6
# recombination_rate is FIXED (not inferred): no signal in pi/dxy/Fst, only in LD (CLAUDE.md 5.4).
DEFAULT_RECOMBINATION_RATE = 2.75e-6

# POPMULT cap: total N ~ 3.33*POPMULT, so 12000 -> ~40k individuals (CHTC-feasible; CLAUDE.md 3).
POPMULT_MAX = 12000

# Define prior distributions using scipy.stats.
# recombination_rate is intentionally ABSENT -- fixed at DEFAULT_RECOMBINATION_RATE (5.4).
prior_distributions = {
    "m": stats.lognorm(s=1.5, scale=np.exp(np.log(0.0001))),  # dispersal-kernel decay (scale), NOT amount
    "total_migration": stats.uniform(loc=0.001, scale=0.3),   # total immigration fraction, U(0.001, 0.301)
    "pop": stats.uniform(loc=2000, scale=POPMULT_MAX - 2000),  # POPMULT in [2000, 12000] ~ [6.7k, 40k] individuals
    "numClusters": stats.randint(1, 4),  # randint(1, 4) gives 1, 2, or 3
    "mutation_rate": stats.lognorm(s=0.5, scale=DEFAULT_MUTATION_RATE),  # nuisance diversity-scaler ~5e-6 (see note above)
}

# --- Fitted-statistic configuration (CLAUDE.md 7) --------------------------------------------
# Subpops sampled too thinly for a usable pairwise Fst are dropped from the FITTED statistics.
# At n<=3 diploids, 46.7% of 2015's pairs return a NEGATIVE Fst (vs 5.3% at 4<=n<=7 and 0% at
# n>=8): those entries scatter around a noise floor rather than measuring differentiation.
# Only 2015 has any subpop this small. Set False to fit every subpop.
EXCLUDE_SMALL_SUBPOPS = True
MIN_SUBPOP_N = 4


# ---------------------------------------------------------------------------
# Feature I/O + helpers (ABC_REFACTOR_PLAN.md 2c)
# ---------------------------------------------------------------------------

def _read_vector(path):
    '''Read a headerless single-column CSV into a 1-D float array. Reads EVERY row (fixes the
    csv.DictReader bug that silently dropped subpop 0 -- CLAUDE.md 5.7).'''
    vals = []
    with open(path, mode='r', newline='', encoding='utf-8') as f:
        for row in csv.reader(f):
            if not row:
                continue
            v = row[0].strip()
            if v != "":
                vals.append(float(v))
    return np.array(vals, dtype=float)


def _read_matrix(path):
    '''Read a headerless square CSV into a 2-D float array; blank cells -> NaN.'''
    matrix = []
    with open(path, mode='r', newline='', encoding='utf-8') as f:
        for row in csv.reader(f):
            if not row:
                continue
            matrix.append([np.nan if v.strip() == "" else float(v) for v in row])
    return np.array(matrix, dtype=float)


_GEO_DIST_CACHE = {}


def _haversine_m(lat1, lon1, lat2, lon2):
    '''Great-circle distance in metres (matches GenerateClusterData.distance formula).'''
    r = 6371000.0
    p = math.pi / 180.0
    a = (0.5 - math.cos((lat2 - lat1) * p) / 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * (1 - math.cos((lon2 - lon1) * p)) / 2)
    return 2 * r * math.asin(math.sqrt(a))


def get_site_geo_distances(year):
    '''Pairwise REAL-SITE geographic distances (metres) for a year's subpops, indexed by
    subpop = specifier-matrix row order -- the same ordering as the pi/Fst/relatedness matrices
    (see GenerateClusterData.assign_genomes_to_clusters_idv_year). Used for BOTH the observed and
    simulated IBD slopes (plan 3.1). Real-site coords are cols 1 (lat), 2 (lon) of the specifier.
    Cached (the specifier files are fixed).'''
    if year in _GEO_DIST_CACHE:
        return _GEO_DIST_CACHE[year]
    coords = np.genfromtxt(Path(f"../data/Genetic_Data/specifier_matrix_{year}.csv"),
                           delimiter=",", usecols=(1, 2))
    lats, lons = coords[:, 0], coords[:, 1]
    n = len(lats)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = _haversine_m(lats[i], lons[i], lats[j], lons[j])
    _GEO_DIST_CACHE[year] = D
    return D


def ibd_slope(fst_matrix, geo_dist):
    '''Rousset isolation-by-distance slope: OLS slope of Fst/(1-Fst) on ln(distance) over all
    off-diagonal pairs. Masks non-finite Fst, Fst>=1, and non-positive distances (e.g. the
    duplicate-site typo -- CLAUDE.md 4). Returns NaN if <2 usable pairs.'''
    fst = np.asarray(fst_matrix, dtype=float)
    n = fst.shape[0]
    xs, ys = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = geo_dist[i, j]
            f = fst[i, j]
            if not np.isfinite(f) or f >= 1.0 or not np.isfinite(d) or d <= 0:
                continue
            xs.append(math.log(d))
            ys.append(f / (1.0 - f))
    if len(xs) < 2:
        return np.nan
    slope, _ = np.polyfit(np.array(xs), np.array(ys), 1)
    return float(slope)


def _offdiag_mean_abs_diff(A, B):
    '''Mean absolute difference over off-diagonal entries (count-normalized within year);
    NaN entries skipped.'''
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n = A.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(np.nanmean(np.abs(A[mask] - B[mask])))


def _pi_log_mean_abs_diff(pi_sim, pi_obs):
    '''Mean absolute difference of log(pi), element-wise (count-normalized); non-finite/non-positive
    entries skipped. Log-space per plan 1.3 (relative error; linearises the theta=4Nmu ridge).'''
    pi_sim = np.asarray(pi_sim, dtype=float)
    pi_obs = np.asarray(pi_obs, dtype=float)
    mask = np.isfinite(pi_sim) & np.isfinite(pi_obs) & (pi_sim > 0) & (pi_obs > 0)
    if not np.any(mask):
        return np.nan
    return float(np.nanmean(np.abs(np.log(pi_sim[mask]) - np.log(pi_obs[mask]))))


_KEEP_MASK_CACHE = {}


def get_keep_mask(year):
    '''Boolean mask over specifier-matrix rows: True = subpop retained in the FITTED statistics.
    Drops subpops with fewer than MIN_SUBPOP_N diploid individuals when EXCLUDE_SMALL_SUBPOPS is
    set. Indexed by specifier row order, so the SAME mask is valid for the observed and the
    simulated matrices (CLAUDE.md 4) -- that is what keeps the comparison element-wise.
    Raises if a specifier site is missing from the popfile rather than silently keeping it.'''
    year = str(year)
    if year in _KEEP_MASK_CACHE:
        return _KEEP_MASK_CACHE[year]

    names = []
    with open(Path(f"../data/Genetic_Data/specifier_matrix_{year}.csv"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                names.append(line.split(",")[0].strip())

    if not EXCLUDE_SMALL_SUBPOPS:
        mask = np.ones(len(names), dtype=bool)
    else:
        counts = {}
        with open(Path(f"../data/Genetic_Data/popFile{year}"), encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    counts[parts[1].strip()] = counts.get(parts[1].strip(), 0) + 1
        missing = [s for s in names if s not in counts]
        if missing:
            raise ValueError(f"{year}: specifier sites absent from popFile{year}: {missing}")
        mask = np.array([counts[s] >= MIN_SUBPOP_N for s in names], dtype=bool)

    _KEEP_MASK_CACHE[year] = mask
    return mask


def model(parameter):
    '''
    The model function that runs the SLiM simulation with the given parameters: 
    1. migration rate (m)
    2. population size (pop)
    3. number of clusters (numClusters)
    4. mutation rate (mutation_rate)
    5. recombination rate (recombination_rate)
    
    :param parameter: This is a dictionary containing the parameters for the simulation.
    '''
    
    #Get the parameters
    m = parameter.get("m", prior_distributions["m"].rvs())
    total_migration = parameter.get("total_migration", prior_distributions["total_migration"].rvs())
    pop = int(np.floor(parameter.get("pop", prior_distributions["pop"].rvs())))
    numClusters = parameter.get("numClusters", prior_distributions["numClusters"].rvs()) * 33  #scale to 33, 66, or 99
    mutation_rate = parameter.get("mutation_rate", DEFAULT_MUTATION_RATE)
    recombination_rate = parameter.get("recombination_rate", DEFAULT_RECOMBINATION_RATE)

    #Run the model - change silent to true for actual runs
    Main.main(num_clusters=numClusters, migration_rates_modifier=m, population_modifier=pop,
              total_migration=total_migration, mutation_rate=mutation_rate, recombination_rate=recombination_rate, silent=True)
    
    
    #Read in the simulated output data (pi vector; dxy, Fst, relatedness matrices)
    outDict = {}
    for year in ["2015", "2019", "2023"]:
        outDict[f"{year}_diversity"] = _read_vector(Path(f"../data/Output_Data/diversities_{year}.csv"))
        outDict[f"{year}_divergence"] = _read_matrix(Path(f"../data/Output_Data/divergences_{year}.csv"))
        outDict[f"{year}_fst"] = _read_matrix(Path(f"../data/Output_Data/fst_{year}.csv"))
        outDict[f"{year}_relatedness"] = _read_matrix(Path(f"../data/Output_Data/relatedness_{year}.csv"))

    return outDict



def calculate_losses(x, x0):
    '''
    Per-statistic distances between observed (x) and simulated (x0) feature sets.

    Each statistic is aggregated within a year by the MEAN over its entries (count-normalized,
    so the 24/17/20-subpop years contribute comparably -- plan 1.5) and then averaged across
    years. pi is compared in LOG space (plan 1.3). There is deliberately NO total_loss: per-plan
    6 the standardization (sigma = 1.4826*MAD from the run set) and the combined distance are
    computed OFFLINE, after the pass (ABC_REFACTOR_PLAN.md 4).

    Returns (all un-standardized):
      - pi_loss     : FITTED  (log-space, element-wise)
      - fst_loss    : FITTED  (off-diagonal)
      - ibd_loss    : DIAGNOSTIC only (|IBD slope difference|)
      - dxy_loss    : DIAGNOSTIC only (off-diagonal)
      - genrel_loss : DIAGNOSTIC only (off-diagonal)

    IBD was demoted from fitted to diagnostic: the OBSERVED slope is indistinguishable from zero
    in all three years (Mantel test, 9999 permutations of site labels: p = 0.64 / 0.15 / 0.99;
    Mantel r = -0.069 / +0.214 / +0.004, sign flipping across years). Fitting a target that is
    noise adds a pure-noise term to the standardized D -- costing acceptance efficiency and
    blurring the identifiable parameters -- and would give `scale` a posterior that is only its
    prior reshaped. Still computed, as a posterior-predictive check.
    '''
    pi_terms, fst_terms, ibd_terms, dxy_terms, genrel_terms = [], [], [], [], []

    for year in ["2015", "2019", "2023"]:
        keep = get_keep_mask(year)
        kk = np.ix_(keep, keep)

        # pi (fitted, log-space, element-wise)
        pi_terms.append(_pi_log_mean_abs_diff(x[f"{year}_diversity"][keep],
                                              x0[f"{year}_diversity"][keep]))

        # Fst (fitted, off-diagonal)
        fst_terms.append(_offdiag_mean_abs_diff(x[f"{year}_fst"][kk], x0[f"{year}_fst"][kk]))

        # IBD slope (diagnostic) -- same real-site distances for observed and simulated
        geo = get_site_geo_distances(year)[kk]
        ibd_terms.append(abs(ibd_slope(x[f"{year}_fst"][kk], geo)
                             - ibd_slope(x0[f"{year}_fst"][kk], geo)))

        # dxy (diagnostic, off-diagonal)
        dxy_terms.append(_offdiag_mean_abs_diff(x[f"{year}_divergence"][kk],
                                                x0[f"{year}_divergence"][kk]))

        # genetic relatedness (diagnostic, off-diagonal) -- deliberately NOT masked. The matrix is
        # centred on the populations present when it was computed, so slicing rows/cols out of it
        # is NOT the same as recomputing on the subset (CLAUDE.md 8.2). Masking here would
        # silently change what the numbers mean; both sides stay full-size instead.
        genrel_terms.append(_offdiag_mean_abs_diff(x[f"{year}_relatedness"],
                                                   x0[f"{year}_relatedness"]))

    return {
        "pi_loss": float(np.nanmean(pi_terms)),
        "fst_loss": float(np.nanmean(fst_terms)),
        "ibd_loss": float(np.nanmean(ibd_terms)),
        "dxy_loss": float(np.nanmean(dxy_terms)),
        "genrel_loss": float(np.nanmean(genrel_terms)),
    }

def getObservedData():
    '''Load empirical features: pi vector, plus dxy / Fst / genetic-relatedness matrices per year.
    Uses _read_vector (which reads every row, fixing the csv.DictReader drop of subpop 0 -- 5.7).'''
    outDict = {}
    for year in ["2015", "2019", "2023"]:
        outDict[f"{year}_diversity"] = _read_vector(Path(f"../data/empiricalStats/averaged_pi_{year}.csv"))
        outDict[f"{year}_divergence"] = _read_matrix(Path(f"../data/empiricalStats/averaged_dxy_{year}.csv"))
        outDict[f"{year}_fst"] = _read_matrix(Path(f"../data/empiricalStats/averaged_fst_{year}.csv"))
        outDict[f"{year}_relatedness"] = _read_matrix(Path(f"../data/empiricalStats/averaged_genRel_{year}.csv"))

    return outDict
    

def sample_prior():
    '''
    Sample parameters from the prior distributions.
    '''
    return {
        "m": prior_distributions["m"].rvs(),
        "total_migration": prior_distributions["total_migration"].rvs(),
        "pop": prior_distributions["pop"].rvs(),
        "numClusters": prior_distributions["numClusters"].rvs(),
        "mutation_rate": prior_distributions["mutation_rate"].rvs(),
        # recombination_rate is fixed (DEFAULT_RECOMBINATION_RATE) -- not sampled (5.4).
    }


def read_parameters_from_csv(csv_path):
    '''
    Read parameter configurations from a CSV file.
    
    Expected CSV columns: m, pop, numClusters, mutation_rate, recombination_rate
    Each row represents one simulation to run.
    
    :param csv_path: Path to the CSV file with parameters
    :return: List of dictionaries, each containing parameters for one simulation
    '''
    parameters_list = []
    
    try:
        with open(csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            if reader.fieldnames is None:
                raise ValueError(f"CSV file {csv_path} is empty or has no headers")
            
            # Validate that all required columns are present (recombination_rate is optional --
            # fixed at DEFAULT_RECOMBINATION_RATE if absent, 5.4).
            required_cols = {"m", "pop", "numClusters", "mutation_rate"}
            csv_cols = set(reader.fieldnames)
            missing_cols = required_cols - csv_cols
            
            if missing_cols:
                raise ValueError(f"CSV file missing required columns: {missing_cols}. "
                                f"Required columns: {required_cols}")
            
            for row_idx, row in enumerate(reader, start=2):  # start=2 because row 1 is header
                try:
                    parameters = {
                        "m": float(row["m"]),
                        # total_migration is optional; default 0.05 so legacy CSVs without the
                        # column still run (plan 2c-iv).
                        "total_migration": float(row["total_migration"]) if row.get("total_migration") not in (None, "") else 0.05,
                        "pop": int(float(row["pop"])),  # Convert to float first to handle scientific notation
                        "numClusters": int(float(row["numClusters"])),
                        "mutation_rate": float(row["mutation_rate"]),
                        "recombination_rate": float(row["recombination_rate"]) if row.get("recombination_rate") not in (None, "") else DEFAULT_RECOMBINATION_RATE
                    }
                    parameters_list.append(parameters)
                except ValueError as e:
                    print(f"Warning: Row {row_idx} in {csv_path} has invalid values: {e}")
                    continue
        
        if not parameters_list:
            raise ValueError(f"No valid parameter configurations found in {csv_path}")
        
        print(f"Loaded {len(parameters_list)} parameter configuration(s) from {csv_path}")
        return parameters_list
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Input CSV file not found: {csv_path}")


def run_sims_from_csv(input_csv, output_csv="../out/abc_results.csv", simToRun=-1):
    '''
    Run ABC simulations with parameters specified in a CSV file.
    Each row in the input CSV represents one simulation.
    Detailed simulation outputs (diversities and divergences) are saved to detailed_sim_results folder.
    
    :param input_csv: Path to the CSV file with input parameters
    :param output_csv: Path to the output CSV file for results
    :param simToRun: Index of the specific simulation to run (if -1, run all)
    '''
    
    try:
        parameters_list = read_parameters_from_csv(input_csv)
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        return
    
    observed_data = getObservedData()
    
    # Determine if we need to write the header.
    # An exists() check alone is not enough: CHTC's run_code.sh pre-creates
    # out/abc_results.csv (so an evicted job fails with a real exit code instead
    # of an errno-2 transfer hold), which made every job see a non-empty path,
    # skip the header, and emit data rows with no column names. Treat a
    # zero-byte file as needing one; a file with rows already in it does not,
    # so appending across iterations still works.
    needs_header = not (Path(output_csv).exists() and Path(output_csv).stat().st_size > 0)
    
    # Create detailed results directory
    output_dir = Path(output_csv).parent
    detailed_results_dir = output_dir / "detailed_sim_results"
    detailed_results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Detailed results will be saved to: {detailed_results_dir}")
    
    # Define CSV columns
    # No per-year columns and no total_loss: the combined standardized distance is built offline
    # (plan 4/6). pi/fst are FITTED; ibd/dxy/genrel are DIAGNOSTIC.
    fieldnames = ["iteration", "m", "total_migration", "pop", "numClusters", "mutation_rate", "recombination_rate",
                  "pi_loss", "fst_loss", "ibd_loss", "dxy_loss", "genrel_loss"]
    
    with open(output_csv, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header if the file is new or empty (see needs_header above)
        if needs_header:
            writer.writeheader()
        
        for iteration, parameters in enumerate(parameters_list):
            if simToRun != -1 and iteration != simToRun:
                continue

            print(
                f"Running iteration {iteration + 1}/{len(parameters_list)} "
                f"with m={parameters['m']:.6g}, total_migration={parameters.get('total_migration', 0.05):.4g}, "
                f"pop={int(np.floor(parameters['pop']))}, "
                f"numClusters={parameters['numClusters'] * 33}, "
                f"mutation_rate={parameters['mutation_rate']:.6g}, "
                f"recombination_rate={parameters.get('recombination_rate', DEFAULT_RECOMBINATION_RATE):.6g}..."
            )
            
            try:
                # Run the model
                simulated_data = model(parameters)
                
                # Calculate losses
                losses = calculate_losses(observed_data, simulated_data)
                
                # Copy detailed results for this iteration
                iteration_dir = detailed_results_dir / f"run{iteration + 1}"
                iteration_dir.mkdir(parents=True, exist_ok=True)
                
                # Detail artifact = raw-feature store for offline standardization (plan 2c-v/4).
                # Copy every raw feature file so post-hoc MAD/sigma has raw values, not just losses.
                for year in ["2015", "2019", "2023"]:
                    for stat in ["diversities", "divergences", "fst", "relatedness"]:
                        src = Path(f"../data/Output_Data/{stat}_{year}.csv")
                        if src.exists():
                            shutil.copy2(src, iteration_dir / f"{stat}_{year}.csv")
                
                # Prepare row for CSV
                row = {
                    "iteration": iteration,
                    "m": parameters["m"],
                    "total_migration": parameters.get("total_migration", 0.05),
                    "pop": parameters["pop"],
                    "numClusters": parameters["numClusters"],
                    "mutation_rate": parameters["mutation_rate"],
                    "recombination_rate": parameters.get("recombination_rate", DEFAULT_RECOMBINATION_RATE),
                    "pi_loss": losses["pi_loss"],
                    "fst_loss": losses["fst_loss"],
                    "ibd_loss": losses["ibd_loss"],
                    "dxy_loss": losses["dxy_loss"],
                    "genrel_loss": losses["genrel_loss"]
                }
                
                # Append to CSV
                writer.writerow(row)
                csvfile.flush()  # Ensure data is written immediately
                
                print(f"  pi={losses['pi_loss']:.4g} fst={losses['fst_loss']:.4g} ibd={losses['ibd_loss']:.4g} "
                      f"dxy={losses['dxy_loss']:.4g} genrel={losses['genrel_loss']:.4g}")
                print(f"  Detailed results saved to: {iteration_dir}")
                
            except Exception as e:
                print(f"  Error in iteration {iteration}: {e}")
                continue
    
    print(f"\n=== CSV-based simulations complete ===")
    print(f"Total iterations: {len(parameters_list)}. Results saved to {output_csv}")
    print(f"Detailed simulation data saved to {detailed_results_dir}")



def run_abc_simulation(num_iterations, output_csv="../out/abc_results.csv"):
    '''
    Run ABC simulations by repeatedly sampling from the prior and computing losses.
    Results are appended to a CSV file.
    
    :param num_iterations: Number of iterations to run
    :param output_csv: Path to the output CSV file
    '''

    observed_data = getObservedData()

    # Determine if we need to write the header.
    # An exists() check alone is not enough: CHTC's run_code.sh pre-creates
    # out/abc_results.csv (so an evicted job fails with a real exit code instead
    # of an errno-2 transfer hold), which made every job see a non-empty path,
    # skip the header, and emit data rows with no column names. Treat a
    # zero-byte file as needing one; a file with rows already in it does not,
    # so appending across iterations still works.
    needs_header = not (Path(output_csv).exists() and Path(output_csv).stat().st_size > 0)

    # Detail artifact dir (raw-feature store for offline standardization; also transferred back
    # from CHTC per the submit file).
    detailed_results_dir = Path(output_csv).parent / "detailed_sim_results"
    detailed_results_dir.mkdir(parents=True, exist_ok=True)

    # Define CSV columns
    # No per-year columns and no total_loss: the combined standardized distance is built offline
    # (plan 4/6). pi/fst are FITTED; ibd/dxy/genrel are DIAGNOSTIC.
    fieldnames = ["iteration", "m", "total_migration", "pop", "numClusters", "mutation_rate", "recombination_rate",
                  "pi_loss", "fst_loss", "ibd_loss", "dxy_loss", "genrel_loss"]

    with open(output_csv, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if the file is new or empty (see needs_header above)
        if needs_header:
            writer.writeheader()

        for iteration in range(num_iterations):
            parameters = sample_prior()
            print(
                f"Running iteration {iteration + 1}/{num_iterations} "
                f"with m={parameters['m']:.6g}, total_migration={parameters.get('total_migration', 0.05):.4g}, "
                f"pop={int(np.floor(parameters['pop']))}, "
                f"numClusters={parameters['numClusters'] * 33}, "
                f"mutation_rate={parameters['mutation_rate']:.6g}, "
                f"recombination_rate={parameters.get('recombination_rate', DEFAULT_RECOMBINATION_RATE):.6g}..."
            )
            
            try:
                # Run the model
                simulated_data = model(parameters)
                
                # Calculate losses
                losses = calculate_losses(observed_data, simulated_data)

                # Copy raw feature files for this trial into the detail artifact (plan 2c-v/4).
                iteration_dir = detailed_results_dir / f"run{iteration + 1}"
                iteration_dir.mkdir(parents=True, exist_ok=True)
                for year in ["2015", "2019", "2023"]:
                    for stat in ["diversities", "divergences", "fst", "relatedness"]:
                        src = Path(f"../data/Output_Data/{stat}_{year}.csv")
                        if src.exists():
                            shutil.copy2(src, iteration_dir / f"{stat}_{year}.csv")

                # Prepare row for CSV
                row = {
                    "iteration": iteration,
                    "m": parameters["m"],
                    "total_migration": parameters.get("total_migration", 0.05),
                    "pop": parameters["pop"],
                    "numClusters": parameters["numClusters"],
                    "mutation_rate": parameters["mutation_rate"],
                    "recombination_rate": parameters.get("recombination_rate", DEFAULT_RECOMBINATION_RATE),
                    "pi_loss": losses["pi_loss"],
                    "fst_loss": losses["fst_loss"],
                    "ibd_loss": losses["ibd_loss"],
                    "dxy_loss": losses["dxy_loss"],
                    "genrel_loss": losses["genrel_loss"]
                }

                # Append to CSV
                writer.writerow(row)
                csvfile.flush()  # Ensure data is written immediately

                print(f"  pi={losses['pi_loss']:.4g} fst={losses['fst_loss']:.4g} ibd={losses['ibd_loss']:.4g} "
                      f"dxy={losses['dxy_loss']:.4g} genrel={losses['genrel_loss']:.4g}")

            except Exception as e:
                print(f"  Error in iteration {iteration}: {e}")
                continue

    print(f"Simulation {iteration + 1} complete. Results saved to {output_csv}")


if __name__ == "__main__":
    # CHTC usage:  python ABCAnalysisNoRedis.py <job_id> [num_trials]
    #   <job_id>     integer identifying this job (e.g. HTCondor $(Process)). It seeds the RNG so
    #                the prior draws are reproducible AND distinct across jobs, and it names the
    #                per-job output file. Give every job a different id.
    #   [num_trials] how many parameter sets to sample from the prior and run (default 100).
    #
    # Each job samples <num_trials> draws from the prior, runs the full pipeline for each, and
    # writes one row per trial to ../out/abc_results.csv (params + per-stat losses, no total_loss),
    # plus per-trial raw features under ../out/detailed_sim_results/. The CHTC submit file transfers
    # both and remaps them per-process (abc_results_<Process>.csv, detailed_sim_results_<Process>).
    # After all jobs finish: concatenate the per-job CSVs and run abc_standardize.py to compute
    # sigma and the combined standardized distance D. See ABC_REFACTOR_PLAN.md §4/§6.
    #
    # (The old CSV-driven path is still available programmatically via run_sims_from_csv().)
    if len(sys.argv) < 2:
        print("Usage: python ABCAnalysisNoRedis.py <job_id> [num_trials]")
        sys.exit(1)

    job_id = int(sys.argv[1])
    num_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    # scipy's <dist>.rvs() draws from numpy's global RNG; seeding it per job makes each job's
    # trials reproducible and (via distinct job_ids) non-overlapping. KMeans stays deterministic
    # regardless (it uses the fixed Main.KMEANS_SEED, not the global RNG).
    np.random.seed(job_id)

    # Fixed filename on purpose: the submit file remaps out/abc_results.csv per-process, so
    # distinct job_ids don't need distinct filenames here (job_id still seeds the draws above).
    output_csv = "../out/abc_results.csv"
    print(f"Job {job_id}: sampling {num_trials} prior-drawn trials -> {output_csv}")
    run_abc_simulation(num_trials, output_csv=output_csv)
