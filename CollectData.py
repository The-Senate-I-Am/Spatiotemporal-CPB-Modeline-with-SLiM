import csv
from collections import defaultdict
import numpy as np
from math import cos, asin, sqrt, pi
import pandas as pd
import matplotlib.pyplot as plt


# Read the CSV file and extract important information into a dictionary
def read_csv(csv_file_path='./data/final_data_for_modeling.csv'):
    # Dictionary to store coordinates for each unique field_fvid (lat, lng, count)
    field_data = defaultdict(list)

    # Read the CSV file and extract important information into field_data
    try:
        with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            count = 0
            for row in reader:
                #if count > 20:
                #    break
                count += 1
                field_fvid = row['field_fvid']
                lat = float(row['lat'])
                lng = float(row['lng'])
                
                # Calculate averages for cpba_count and cpbl_count
                cpba_count = float(row['cpba_count'])
                cpbl_count = float(row['cpbl_count'])
                avg_count = (cpba_count + cpbl_count) / 2
                
                # Add to field_coordinates
                field_data[field_fvid] = (lat, lng, avg_count)
                
    except FileNotFoundError:
        print(f"Error: File not found at {csv_file_path}")
    except KeyError as e:
        print(f"Error: Missing expected column in CSV file: {e}")
    except ValueError as e:
        print(f"Error: Invalid data format in CSV file: {e}")
        
    return field_data




# Calculate distances between field coordinates and save to a distance matrix
def create_distance_matrix(field_data, cutoff=10000):
    # Calculate distances between field coordinates and save to a distance matrix

    # Function to calculate distance between two coordinates in km using Haversine formula
    def distance(lat1, lon1, lat2, lon2):
        r = 6371 # km
        p = pi / 180

        a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
        return 2 * r * asin(sqrt(a))

    grid = np.eye(len(field_data)+1, dtype=int)
    fvids = list(field_data.keys())
    for i in range(len(field_data)):
        grid[i+1][0] = fvids[i]
        grid[0][i+1] = fvids[i]

    for i in range(len(field_data)):
        for j in range(i, len(field_data)):
            lat1 = field_data[fvids[i]][0]
            lon1 = field_data[fvids[i]][1]
            lat2 = field_data[fvids[j]][0]
            lon2 = field_data[fvids[j]][1]
        
            dist = int(1000* distance(lat1, lon1, lat2, lon2)) # Convert km to meters for more precision
            if (dist <= cutoff): # Only store distances less than or equal to cutoff
                grid[i+1][j+1] = dist

    np.savetxt('./data/distances.csv', grid, delimiter = ",", fmt='%i')



# Plotting the coordinates using matplotlib to visualize the field locations
def plot_coordinates(field_data, output_path='./out/field_locations.png'):
    # Prepare data for plotting
    plot_data = {'Latitude': [], 'Longitude': []}
    for _, (lat, lng, _) in field_data.items():
        plot_data['Latitude'].append(lat)
        plot_data['Longitude'].append(lng)

    # Convert to a format suitable for plotting
    df = pd.DataFrame(plot_data)

    plt.scatter(x=df['Longitude'], y=df['Latitude'])
    #plt.show()
    # Save the plot to a PNG file
    output_plot_path = './out/field_locations.png'
    plt.savefig(output_plot_path, format='png')
    



# Query the user for their preferences
calculate_distance_matrix = input("Calculate the distance matrix? (y/n): ").strip().lower() == 'y'
plot_coordinates = input("Plot the coordinates? (y/n): ").strip().lower() == 'y'

CUTOFF = 10000  # Default cutoff distance in meters
plot_coordinates_path = './out/field_locations.png'



field_data = read_csv('./data/final_data_for_modeling.csv')
print("Read data from CSV file successfully.")

if (calculate_distance_matrix):
    create_distance_matrix(field_data, cutoff=CUTOFF)
    print("Distance matrix saved to './data/distances.csv'.")

if (plot_coordinates):
    plot_coordinates(field_data, output_path=plot_coordinates_path)
    print(f"Plot saved to '{plot_coordinates_path}'.")