import csv
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

import pyabc
from pyabc.transition import MultivariateNormalTransition
from pyabc.sampler import RedisEvalParallelSampler

import Main
from ABCAnalysisNoRedis import DEFAULT_MUTATION_RATE, DEFAULT_RECOMBINATION_RATE

pyabc.settings.set_figure_params('pyabc')  # for beautified plots

prior = pyabc.Distribution(
    m=pyabc.RV("lognorm", np.log(0.0001), 1.5),
    pop=pyabc.RV("expon", loc=2000, scale=50000),
    numClusters=pyabc.RV("randint", 1, 3)
)



def model(parameter):
    '''
    The model function that runs the SLiM simulation with the given parameters: 
    1. migration rate (m)
    2. population size (pop)
    3. number of clusters (numClusters)   
    
    :param parameter: This is a dictionary containing the parameters for the simulation.
    '''
    
    #Get the parameters
    m = parameter["m"]
    pop = parameter["pop"]
    numClusters = parameter["numClusters"] * 33  #scale to 33, 66, or 99
    
    #Run the model
    # mutation_rate is now required by Main.main() (no silent wrong-scale default; CLAUDE.md 10).
    Main.main(num_clusters=numClusters, migration_rates_modifier=m, population_modifier=pop,
              mutation_rate=DEFAULT_MUTATION_RATE, recombination_rate=DEFAULT_RECOMBINATION_RATE,
              silent=True)
    
    
    #Read in the output data
    outDict = {}
    
    for year in ["2015", "2019", "2023"]:
        with open(Path(f"../data/Output_data/diversities_{year}.csv"), mode='r', newline='', encoding='utf-8') as csvfile:
            div2015 = csv.DictReader(csvfile)
            diversities_list = []
            for row in div2015:
                value = next(iter(row.values()))
                if value is not None and value.strip() != "":
                    diversities_list.append(float(value.strip()))
            diversities = np.array(diversities_list)
            outDict[f"{year}_diversity"] = diversities
        
        with open(Path(f"../data/Output_data/divergences_{year}.csv"), mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            matrix = []
            for row in reader:
                if not row:
                    continue
                row_vals = []
                for val in row:
                    v = val.strip()
                    if v == "":
                        row_vals.append(np.nan)
                    else:
                        row_vals.append(float(v))
                matrix.append(row_vals)
            divergences = np.array(matrix, dtype=float)
            outDict[f"{year}_divergence"] = divergences
        

    return outDict



def distance(x, x0):
    '''
    Distance function for comparing simulated and observed data.
    x is observed, x0 is simulated.
    Pi is nucleotide diversity given in the form of a list of values for each population.
    dxy is genetic differentiation given in the form of a matrix of values between populations.
    
    x should be a dictionary with keys "2015_diversity", "2015_divergence", "2019_diversity", "2019_divergence",
    "2023_diversity", and "2023_divergence".
    
    Each of these values is an array of three of these values, one for each year (2015, 2019, 2023).
    
    Returns a single distance value.
    '''
    total_distance = 0
    
    for year in ["2015", "2019", "2023"]:
        diversity_key = f"{year}_diversity"
        divergence_key = f"{year}_divergence"

        # Pi distance
        for i in range(len(x[diversity_key])):
            pi_distance = abs(x[diversity_key][i] - x0[diversity_key][i])
            pi_distance *= len(x[diversity_key]) #weight pi distance equally to fst distance
            total_distance += pi_distance
        # Fst distance
        for i in range(len(x[divergence_key])):
            for k in range(len(x[divergence_key])):
                if i != k:
                    fst_distance = abs(x[divergence_key][i][k] - x0[divergence_key][i][k])
                    total_distance += fst_distance
                    
    return total_distance

def getObservedData():
    outDict = {}
    
    for year in ["2015", "2019", "2023"]:
        path = f"../data/empiricalStats/averaged_pi_{year}.csv"
        with open(Path(path), mode='r', newline='', encoding='utf-8') as csvfile:
            div2015 = csv.DictReader(csvfile)
            diversities_list = []
            for row in div2015:
                value = next(iter(row.values()))
                if value is not None and value.strip() != "":
                    diversities_list.append(float(value.strip()))
            diversities = np.array(diversities_list)
            outDict[f"{year}_diversity"] = diversities
        
        path = f"../data/empiricalStats/averaged_dxy_{year}.csv"
        with open(Path(path), mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            matrix = []
            for row in reader:
                if not row:
                    continue
                row_vals = []
                for val in row:
                    v = val.strip()
                    if v == "":
                        row_vals.append(np.nan)
                    else:
                        row_vals.append(float(v))
                matrix.append(row_vals)
            divergences = np.array(matrix, dtype=float)
            outDict[f"{year}_divergence"] = divergences
            
    return outDict
    

#with redis
redis_sampler = RedisEvalParallelSampler(host="111.111.111.111", port=6379)
abc = pyabc.ABCSMC(model, prior, distance, population_size=500, transitions=MultivariateNormalTransition(), sampler=redis_sampler)

#without redis
#abc = pyabc.ABCSMC(model, prior, distance, population_size=500, transitions=MultivariateNormalTransition())

db_path = os.path.join(tempfile.gettempdir(), "test.db")

observation = getObservedData()
    
abc.new("sqlite:///" + db_path, observation)

history = abc.run(minimum_epsilon=2, max_nr_populations=5)
df, weights = history.get_distribution()
df.to_csv("posterior_samples.csv", index=False)

# df, w = history.get_distribution()
# posterior_mean = (df * w[:, None]).sum()
# print("Posterior mean:", posterior_mean)