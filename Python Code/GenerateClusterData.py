import numpy as np
import pandas as pd
import random
from math import pi, cos, asin, sqrt
from sklearn.cluster import KMeans
import DataWrappers

# Cluster the coordinates using KMeans and add cluster labels to the field data    
def cluster_coordinates(field_data, n_clusters=5, iters=2000, random_state=random.randint(0, 1000)):
    # Prepare data for clustering
    coordinates = np.array([[field.latitude, field.longitude] for field in field_data.values()])
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, max_iter=iters, random_state=random_state, n_init='auto')
    kmeans.fit(coordinates)
    
    # Add cluster labels to the field data
    for i, fvid in enumerate(field_data.keys()):
        field_data[fvid].set_cluster(kmeans.labels_[i])
    
    return field_data


# This function is used to take field data that has been run through cluster_coordinates and create a new list
# that contains averaged data for each cluster such as location and count
def populate_cluster_objects(field_data, estimate_data=True):
    # Start by creating a list of lists of lists which sorts each node into its respective cluster. 
    # Each cluster has a list of latitudes, longitudes, average counts, and field_fvids which will be operated on to create generalized cluster data
    clusters = [None] * (1+max(field_data[fvid].cluster for fvid in field_data.keys()))  # Create a list of None with length equal to the number of clusters
    
    for fvid in field_data.keys():
        cluster_id = field_data[fvid].cluster
        
        #print(len(clusters), cluster_id)
        if clusters[cluster_id] is None:
            clusters[cluster_id] = DataWrappers.Cluster(cluster_id)
        clusters[cluster_id].add_field(field_data[fvid])
    
    # Calculate the yearly data and coordinates for each cluster based on inputted fields
    for cluster in clusters:
        cluster.calculate_coordinates()
        cluster.average_data()
            
    return clusters



# This function takes a list of clusters and calculates the distance between each pair of clusters, storing the results in a distance matrix 
#The csv file is outputted to the specified path and returned by the function.
def create_cluster_distance_matrix(clusters, output_path='../data/cluster_distances.csv'):
    # Function to calculate distance between two coordinates in km using Haversine formula
    def distance(lat1, lon1, lat2, lon2):
        r = 6371  # km
        p = pi / 180

        a = 0.5 - cos((lat2 - lat1) * p) / 2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
        return int(2 * r * asin(sqrt(a)) * 1000)  # Convert km to meters for more precision

    # Create a distance matrix with cluster IDs as both row and column headers
    grid = np.eye(len(clusters) + 1, dtype=int)
    for i in range(len(clusters)):
        grid[i + 1][0] = clusters[i].cluster_id 
        grid[0][i + 1] = clusters[i].cluster_id 
    grid[0][0] = 0

    # Calculate distances and fill the distance matrix storing data only if dist is <= cutoff
    for i in range(len(clusters)):
        for j in range(i, len(clusters)):
            lat1 = clusters[i].latitude
            lon1 = clusters[i].longitude
            lat2 = clusters[j].latitude
            lon2 = clusters[j].longitude
        
            dist = distance(lat1, lon1, lat2, lon2)  # Convert km to meters for more precision
            grid[i + 1][j + 1] = dist
            grid[j + 1][i + 1] = dist  # Ensure symmetry in the distance matrix

    np.savetxt(output_path, grid, delimiter=",", fmt='%i')
    return grid


# This function takes a list of cluster objects and saves their data to a CSV file
def cluster_data_to_csv(clusters, output_path='../data/cluster_data.csv'):
    # Create a DataFrame to hold the cluster data
    data = {
        'Cluster ID': [],
        'Latitude': [],
        'Longitude': [],
        'Average Count': [],
        'Average GDD': []
    }
    
    for cluster in clusters:
        data['Cluster ID'].append(cluster.cluster_id)
        data['Latitude'].append(cluster.latitude)
        data['Longitude'].append(cluster.longitude)
        data['Average Count'].append(cluster.data[0])  # Assuming first element is average count
        data['Average GDD'].append(cluster.data[1])  # Assuming second element is average GDD
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)