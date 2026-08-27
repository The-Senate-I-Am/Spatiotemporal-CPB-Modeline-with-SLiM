import csv
import math
import numpy as np
import sys
import shutil

from pathlib import Path
from scipy import stats

# Main is imported lazily inside model() so this module stays importable without the simulation
# stack; diagnostics/*.py reuse the readers and get_keep_mask (CLAUDE.md 9).

# NOT a biological mutation rate. Half of a calibration constant -- only theta = 4*Ne_anc*mu is
# meaningful, so report theta, never mu. Calibrated at POPMULT=5000 (CLAUDE.md 6.1.1).
DEFAULT_MUTATION_RATE = 4.646e-7
# Fixed, not inferred: no signal outside LD (CLAUDE.md 6.3).
DEFAULT_RECOMBINATION_RATE = 2.75e-6

# Total N ~ 3.33*POPMULT. Raised 12000 -> 25000 on 2026-08-26 (CLAUDE.md 6.7).
# ~44 GB and ~1.9 h per trial at the ceiling -- size CHTC requests from CLAUDE.md 3.1.
POPMULT_MAX = 25000

# recombination_rate is intentionally absent -- fixed at DEFAULT_RECOMBINATION_RATE.
prior_distributions = {
    "m": stats.lognorm(s=1.5, scale=np.exp(np.log(0.0001))),   # kernel decay; unidentifiable (7.1)
    "total_migration": stats.uniform(loc=0.001, scale=0.3),     # U(0.001, 0.301)
    "pop": stats.uniform(loc=2000, scale=POPMULT_MAX - 2000),   # POPMULT ~ U(2000, 25000)
    "numClusters": stats.randint(1, 4),                         # 1, 2 or 3; scaled x33 in model()
    # Nuisance parameter. s is the fractional spread, tightened 0.5 -> 0.05 -> 0.02: anything
    # wider lets the mu draw rather than POPMULT dominate pi_loss (CLAUDE.md 7.2.1).
    "mutation_rate": stats.lognorm(s=0.02, scale=DEFAULT_MUTATION_RATE),
}

# Subpops too thinly sampled for a usable pairwise Fst are dropped from the FITTED statistics --
# at n<=3, 46.7% of 2015's pairs return a negative Fst. Only 2015 is affected (CLAUDE.md 7.0).
EXCLUDE_SMALL_SUBPOPS = True
MIN_SUBPOP_N = 4


# ---------------------------------------------------------------------------
# Feature I/O + helpers
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
    simulated IBD slopes. Real-site coords are cols 1 (lat), 2 (lon) of the specifier.
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
    entries skipped. Log-space gives relative error and linearises the theta=4Nmu ridge.'''
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
    import Main   # lazy: see the note beside the imports at the top of this file
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

    Each statistic is averaged over its entries within a year (so the 24/17/20-subpop years
    contribute comparably), then across years. pi is compared in LOG space. There is deliberately
    NO total_loss -- abc_standardize.py builds the combined distance offline (CLAUDE.md 7).

    Returns (all un-standardized):
      - pi_loss     : FITTED  (log-space, element-wise)
      - fst_loss    : FITTED  (off-diagonal)
      - ibd_loss    : DIAGNOSTIC only (|IBD slope difference|)
      - dxy_loss    : DIAGNOSTIC only (off-diagonal)
      - genrel_loss : DIAGNOSTIC only (off-diagonal)

    IBD is diagnostic, not fitted: the observed slope is indistinguishable from zero in all three
    years (CLAUDE.md 7.1).
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

        # Deliberately NOT masked: relatedness is centred on the populations present when it was
        # computed, so slicing it is not the same as recomputing on the subset (CLAUDE.md 8#2).
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
                        # column still run.
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
    # CHTC's run_code.sh pre-creates this file, so exists() alone would skip the header.
    # Treat a zero-byte file as needing one.
    needs_header = not (Path(output_csv).exists() and Path(output_csv).stat().st_size > 0)
    
    # Create detailed results directory
    output_dir = Path(output_csv).parent
    detailed_results_dir = output_dir / "detailed_sim_results"
    detailed_results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Detailed results will be saved to: {detailed_results_dir}")
    
    # Define CSV columns
    # No total_loss: the combined standardized distance is built offline by abc_standardize.py.
    # pi/fst are FITTED; ibd/dxy/genrel are DIAGNOSTIC.
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
                
                # Keep raw features so offline sigma has values, not just losses.
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
    # CHTC's run_code.sh pre-creates this file, so exists() alone would skip the header.
    # Treat a zero-byte file as needing one.
    needs_header = not (Path(output_csv).exists() and Path(output_csv).stat().st_size > 0)

    # Raw-feature store for offline standardization; transferred back from CHTC.
    detailed_results_dir = Path(output_csv).parent / "detailed_sim_results"
    detailed_results_dir.mkdir(parents=True, exist_ok=True)

    # Define CSV columns
    # No total_loss: the combined standardized distance is built offline by abc_standardize.py.
    # pi/fst are FITTED; ibd/dxy/genrel are DIAGNOSTIC.
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

                # Keep this trial's raw features for offline standardization.
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
    # Usage: python ABCAnalysisNoRedis.py <job_id> [num_trials]
    # job_id seeds the RNG and must be DISTINCT per job -- repeats re-draw identical parameters.
    # Writes one row per trial to ../out/abc_results.csv plus raw features under
    # ../out/detailed_sim_results/. Afterwards, concatenate the per-job CSVs and run
    # abc_standardize.py. (run_sims_from_csv() is the older CSV-driven path.)
    if len(sys.argv) < 2:
        print("Usage: python ABCAnalysisNoRedis.py <job_id> [num_trials]")
        sys.exit(1)

    job_id = int(sys.argv[1])
    num_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    # scipy .rvs() draws from numpy's global RNG, so this makes each job's trials reproducible
    # and distinct. KMeans is unaffected -- it uses the fixed Main.KMEANS_SEED.
    np.random.seed(job_id)

    # Fixed filename: the submit file remaps it per-process.
    output_csv = "../out/abc_results.csv"
    print(f"Job {job_id}: sampling {num_trials} prior-drawn trials -> {output_csv}")
    run_abc_simulation(num_trials, output_csv=output_csv)
