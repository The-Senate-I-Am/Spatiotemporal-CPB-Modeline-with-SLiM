import csv
import numpy as np
from math import cos, asin, sqrt, pi
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random

import DataWrappers

# Read the CSV file and extract important information into a dictionary
def read_csv(csv_file_path='./data/final_data_for_modeling.csv'):
    # Dictionary to store field objects for each unique field_fvid
    field_data = dict()

    # Read the CSV file and extract important information into field_data
    try:
        with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                field_fvid = row['field_fvid']
                lat = float(row['lat'])
                lng = float(row['lng'])
                year = int(row['year'])
                gdd = float(row['gdd'])
                
                # Calculate averages for cpba_count and cpbl_count
                cpba_count = float(row['cpba_count'])
                cpbl_count = float(row['cpbl_count'])
                avg_count = (cpba_count + cpbl_count) / 2
                
                # Add to field_data
                if (field_fvid not in field_data):
                    field_data[field_fvid] = DataWrappers.Field(field_fvid, lat, lng)
                    field_data[field_fvid].add_data_point(year, avg_count, gdd) 
                else:
                    field_data[field_fvid].add_data_point(year, avg_count, gdd)  
                
    except FileNotFoundError:
        print(f"Error: File not found at {csv_file_path}")
    except KeyError as e:
        print(f"Error: Missing expected column in CSV file: {e}")
    except ValueError as e:
        print(f"Error: Invalid data format in CSV file: {e}")
        
    return field_data




# Calculate distances between each field coordinate and save to a distance matrix
def create_distance_matrix(field_data, cutoff=10000):
    # Function to calculate distance between two coordinates in km using Haversine formula
    def distance(lat1, lon1, lat2, lon2):
        r = 6371 # km
        p = pi / 180

        a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
        return int(2 * r * asin(sqrt(a)) * 1000)  # Convert km to meters for more precision

    # Create a distance matrix with field_fvids as both row and column headers
    grid = np.eye(len(field_data)+1, dtype=int)
    fvids = list(field_data.keys())
    for i in range(len(field_data)):
        grid[i+1][0] = fvids[i] 
        grid[0][i+1] = fvids[i]

    # Calculate distances and fill the distance matrix storing data only if dist is <= cutoff
    for i in range(len(fvids)):
        for j in range(i, len(fvids)):
            lat1 = field_data[fvids[i]].latitude
            lon1 = field_data[fvids[i]].longitude
            lat2 = field_data[fvids[j]].latitude
            lon2 = field_data[fvids[j]].longitude
        
            dist = distance(lat1, lon1, lat2, lon2) # Convert km to meters for more precision
            if (dist <= cutoff): # Only store distances less than or equal to cutoff
                grid[i+1][j+1] = dist

    np.savetxt('./data/distances.csv', grid, delimiter = ",", fmt='%i')



# Plotting the coordinates using matplotlib to visualize the field locations
def plot_coordinates(field_data, output_path='./out/field_locations.png'):
    # Prepare data for plotting
    plot_data = {'Latitude': [], 'Longitude': [], 'Cluster': []}
    for field in field_data.values():
        plot_data['Latitude'].append(field.latitude)
        plot_data['Longitude'].append(field.longitude)
        # Use the fourth field (cluster) if it exists, otherwise default to None
        plot_data['Cluster'].append(field.cluster if field.cluster != -1 else None)

    # Convert to a format suitable for plotting
    df = pd.DataFrame(plot_data)

    # If 'Cluster' column has valid data, use it for coloring
    if df['Cluster'].notnull().all():
        plt.scatter(x=df['Longitude'], y=df['Latitude'], c=df['Cluster'], cmap='viridis', s=50)
    else:
        plt.scatter(x=df['Longitude'], y=df['Latitude'], color='blue', s=50)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Field Locations')
    #plt.colorbar(label='Cluster') if df['Cluster'].notnull().all() else None

    # Save the plot to a PNG file
    plt.savefig(output_path, format='png')
    plt.close()
    
    
# Cluster the coordinates using KMeans and add cluster labels to the field data    
def cluster_coordinates(field_data, n_clusters=5, iters=2000, random_state=random.randint(0, 1000)):
    # Prepare data for clustering
    coordinates = np.array([[field.latitude, field.longitude] for field in field_data.values()])
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, max_iter=iters, random_state=random_state)
    kmeans.fit(coordinates)
    
    # Add cluster labels to the field data
    for i, fvid in enumerate(field_data.keys()):
        field_data[fvid].set_cluster(kmeans.labels_[i])
    
    return field_data
    

# This function is used to take field data that has been run through cluster_coordinates and create a new list
# that contains averaged data for each cluster such as location and count
def create_cluster_data(field_data, estimate_data=True):
    # Start by creating a list of lists of lists which sorts each node into its respective cluster. 
    # Each cluster has a list of latitudes, longitudes, average counts, and field_fvids which will be operated on to create generalized cluster data
    clusters = []
    for fvid in field_data.keys():
        cluster_id = field_data[fvid].cluster
        
        if clusters[cluster_id] is None:
            clusters[cluster_id] = DataWrappers.Cluster(cluster_id)
        clusters[cluster_id].add_field(field_data[fvid])
    
    # Calculate the yearly data and coordinates for each cluster based on inputted fields
    for cluster in clusters:
        cluster.calculate_coordinates()
        cluster.fill_data(year_start=2014, year_end=2024)  # Fill the cluster's data with averages for each year
        if estimate_data:
            cluster.estimate_data(year_start=2014, year_end=2024)  # Estimate data if required
            
    return clusters
        
        



# Query the user for their preferences
create_distance_matrix_in = input("Calculate the distance matrix? (y/n): ").strip().lower() == 'y'
plot_coordinates_in = input("Plot the coordinates? (y/n): ").strip().lower() == 'y'
clustered_in = False
num_clusters = 0
if (plot_coordinates_in):
    clustered_in = input("Cluster the coordinates before plotting? (y/n): ").strip().lower() == 'y'
    num_clusters = int(input("Enter the number of clusters (default 50): ").strip() or 50)

CUTOFF = 10000  # Cutoff distance in meters for the distance matrix
plot_coordinates_path_unclustered = './out/field_locations.png'
plot_coordinates_path_clustered = './out/field_locations_clustered.png'


# Read the CSV file and extract important information into field_data
field_data = read_csv('./data/final_data_for_modeling.csv')
print("Read data from CSV file successfully.")

# Perform actions based on user preferences
if (create_distance_matrix_in):
    create_distance_matrix(field_data, cutoff=CUTOFF)
    print("Distance matrix saved to './data/distances.csv'.")

if (plot_coordinates_in):
    if clustered_in:
        cluster_coordinates(field_data, n_clusters=num_clusters, iters=2000, random_state=random.randint(0, 1000))
        print("Coordinates clustered successfully.")
        plot_coordinates(field_data, output_path=plot_coordinates_path_clustered)
        print(f"Plot saved to '{plot_coordinates_path_clustered}'.")
    else:
        plot_coordinates(field_data, output_path=plot_coordinates_path_unclustered)
        print(f"Plot saved to '{plot_coordinates_path_unclustered}'.")
