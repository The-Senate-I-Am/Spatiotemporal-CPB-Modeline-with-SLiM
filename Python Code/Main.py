import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import DataWrappers
import CollectData
import GenerateSimulationParams
import GenerateClusterData

#WARNING: don't run this file in VSCode. Run it in the terminal instead.

def main():
    #Start by reading the data from final_data_for_modeling.csv
    field_data = CollectData.read_csv('../data/final_data_for_modeling.csv')
    
    #Create clusters from the field data
    num_clusters = int(input("Enter the number of clusters (default 50): ").strip() or 50)
    
    #Cluster the coordinates using KMeans
    GenerateClusterData.cluster_coordinates(field_data, n_clusters=num_clusters, iters=2000, random_state=random.randint(0, 1000))
    
    #Put the data for clusters into a list of Cluster objects
    clusters = GenerateClusterData.populate_cluster_objects(field_data, estimate_data=True)

    #Generate a distance matrix for the clusters
    distances = GenerateClusterData.create_cluster_distance_matrix(clusters, output_path='../data/cluster_distances.csv')   
     
    #Save the cluster data to a CSV file
    GenerateClusterData.cluster_data_to_csv(clusters, output_path='../data/cluster_data.csv')
    
    #Generate migration rates based on the cluster distance matrix
    migration_rates = GenerateSimulationParams.determine_migration_rates(distances, modifier=10000, output_path='../data/migration_rates.csv')
    

    
main()