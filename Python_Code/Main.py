import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from pathlib import Path

import DataWrappers
import CollectData
import GenerateSimulationParams
import GenerateClusterData
import AnalyzeTreeSeq

import subprocess

import warnings
import platform
import sys

#WARNING: don't run this file in VSCode. Run it in the terminal instead.

# Fixed KMeans seed so cluster identity (and the subpop->coordinate mapping the IBD slope and
# every summary statistic depend on) is stable across ABC iterations. See CLAUDE.md 5.6.
KMEANS_SEED = 42

def main(num_clusters, migration_rates_modifier, population_modifier, total_migration=0.05, mutation_rate=5e-6, recombination_rate=2.75e-6, ancestral_Ne=6700, silent=False):
    #for cleanliness
    warnings.filterwarnings("ignore")
    
    #Query for mutation rate
    #mutation_rate = 2.1e-9 #float(input("Enter the mutation rate (default 1e-7): ").strip() or 2.1e-9)

    #Query for recombination rate
    #recombination_rate = 2.75e-6 #float(input("Enter the recombination rate (default 1e-8): ").strip() or 2.75e-6)
        
    #Start by reading the data from final_data_for_modeling.csv
    if not silent:
        print("Setting up data for simulations...")
    field_data = CollectData.read_csv(Path('../data/final_data_for_modeling.csv'))
    
    #Cluster the coordinates using KMeans
    GenerateClusterData.cluster_coordinates(field_data, n_clusters=num_clusters, iters=2000, random_state=KMEANS_SEED)
    
    #Put the data for clusters into a list of Cluster objects
    clusters = GenerateClusterData.populate_cluster_objects(field_data, estimate_data=True)
    
    #Assign genomes to clusters based on the specifier matrix for a certain year
    GenerateClusterData.assign_genomes_to_clusters(clusters)
    
    #Generate a distance matrix for the clusters
    distances = GenerateClusterData.create_cluster_distance_matrix(clusters, output_path=Path('../data/cluster_distances.csv'))   
     
    #Save the cluster data to a CSV file
    GenerateClusterData.cluster_data_to_csv(clusters, output_path=Path('../data/cluster_data.csv'))
        
    #Generate migration rates based on the cluster distance matrix
    GenerateSimulationParams.determine_migration_rates(distances, total_migration=total_migration, scale=migration_rates_modifier, output_path=Path('../data/migration_rates.csv'))
    
    #Run the SLiM simulation to create the tree sequence
    if not silent:
        print("Running SLiM simulation...")
    
    #Run the appropriate SLiM script based on the operating system, passing in the population modifier as a parameter
    if platform.system() == "Windows":
        subprocess.run(['slim', '-l', '0', '-d', f'POPMULT={population_modifier}', str(Path('../SLiM_Code/CPBSampleSimWin.slim'))])
    elif platform.system() == "Linux":
        subprocess.run(['slim', '-l', '0', '-d', f'POPMULT={population_modifier}', str(Path('../SLiM_Code/CPBSampleSimLinux.slim'))])
    else:
        raise OSError(f"Unsupported operating system: {platform.system()}")
    
    #Does recapitation and mutation addition, then gets diversity and divergence statistics
    if not silent:
        print("Recapitating tree sequence...")
    AnalyzeTreeSeq.analyze_tree_sequence(mutation_rate=mutation_rate, recombination_rate=recombination_rate, ancestral_Ne=ancestral_Ne)
    if not silent:
        print("Successfully generated diversity and divergence statistics from tree sequence.")
    
    
    

if __name__ == "__main__":
    #Ask for input on number of clusters
    num_clusters = int(input("Enter the number of clusters (default 99): ").strip() or 99)
    
    #Query for a migration rates modifier
    migration_rates_modifier = float(input("Enter the migration rates modifier (default 0.0001): ").strip() or 0.0001)
    
    #Query for population modifier
    population_modifier = float(input("Enter the total population size (default 10000): ").strip() or 10000)
    
    main(num_clusters, migration_rates_modifier, population_modifier)
