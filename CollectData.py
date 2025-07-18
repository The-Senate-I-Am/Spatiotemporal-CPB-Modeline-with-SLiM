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
        plt.scatter(x=df['Longitude'], y=df['Latitude'], c=df['Cluster'], cmap='viridis', s=30)
    else:
        plt.scatter(x=df['Longitude'], y=df['Latitude'], color='blue', s=30)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Field Locations')
    #plt.colorbar(label='Cluster') if df['Cluster'].notnull().all() else None

    # Save the plot to a PNG file
    plt.savefig(output_path, format='png')
    plt.close()



# This function takes a list of clusters and plots their coordinates on a scatter plot, saving the plot to a PNG file
def plot_cluster_coordinates(clusters, output_path='./out/cluster_field_locations.png'):
    # Prepare data for plotting
    plot_data = {'Latitude': [], 'Longitude': [], 'Cluster ID': []}
    for cluster in clusters:
        plot_data['Latitude'].append(cluster.latitude)
        plot_data['Longitude'].append(cluster.longitude)
        plot_data['Cluster ID'].append(cluster.cluster_id)

    # Convert to a format suitable for plotting
    df = pd.DataFrame(plot_data)

    # Plot the coordinates with cluster IDs
    plt.scatter(x=df['Longitude'], y=df['Latitude'], color='blue', s=50)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Cluster Field Locations')

    # Save the plot to a PNG file
    plt.savefig(output_path, format='png')
    plt.close()

    

# field_data = read_csv('./data/final_data_for_modeling.csv')
# #create_distance_matrix(field_data, cutoff=CUTOFF)
# num_clusters = int(input("Enter the number of clusters (default 50): ").strip() or 50)
# cluster_coordinates(field_data, n_clusters=num_clusters, iters=2000, random_state=random.randint(0, 1000))
# #plot_coordinates(field_data, output_path=plot_coordinates_path_clustered)


# clusters = create_cluster_data(field_data, estimate_data=True)
# plot_cluster_coordinates(clusters, output_path='./out/cluster_field_locations.png')
# create_cluster_distance_matrix(clusters, output_path='./data/cluster_distances.csv')

# cluster_data_to_csv(clusters, output_path='./data/cluster_data.csv')