import pandas as pd
from collections import Counter
import csv

def create_specifier_matrix(year, genetic_data="../data/Genetic_Data/CPB_genetic_metadata.csv", output_path="../data/Genetic_Data/specifier_matrix.csv"):
    """
    Create a specifier matrix for the given year from genetic data. 
    This matrix will have rows with population names, latitude, longitude, and indices of genetic data.
    
    Parameters:
    year (int): The year for which to create the specifier matrix.
    genetic_data (str): Path to the genetic data CSV file.
    output_path (str): Path to save the specifier matrix CSV file.
    
    Returns:
    None
    """
    # Read the genetic data
    df = pd.read_csv(genetic_data)
    
    matrix = []
    currPop = None
    
    for index, row in df.iterrows():
        # You can access each row's data using 'row'
        if row['Year'] != year:
            continue
        # print(row)
        # print("----------------------------")
        if currPop is None or currPop != row['Population']:
            matrix.append([row['Population'], row['Latitude'], row['Longitude']])
            currPop = row['Population']
        else:
            matrix[-1].append(index)
        
        #print(matrix)
        # # Convert matrix to DataFrame and save to CSV
        specifier_df = pd.DataFrame(matrix)
        specifier_df.to_csv(output_path, index=False, header=False)

    

def generate_Ref_FASTA_from_genolike(cutoff, filePath="../../../Data for modeling/cpbWGS_genolike_chr9.cpbWGS_genolike_chr9.beagle.gz.phased", output_path_fasta="../data/Genetic_Data/refFull.fasta", output_path_ids="../data/Genetic_Data/FastaIDs.csv"):
    """
    Generate a reference FASTA file from genetic data.
    This function reads the genetic data and creates a FASTA file.
    
    Parameters:
    cutoff (int): The cutoff value for filtering genetic data. If the percentage of similar base pairs is above this cutoff,
            it is not included in the FASTA file.
    filePath (str): Path to the genetic data file. This file is not in the github repository and should be provided separately.
    
    Returns:
    None
    """
    with open(filePath, 'r') as f:
        #lines = f.readlines()
        lines = [next(f) for _ in range(10000)] #TODO: remove this line and uncomment the previous line to read the full file
        
    print("read done")

    matrix = [line.strip().split() for line in lines if line.strip()]
    
    # Remove the first row and first column from the matrix which just stores headers
    matrix = [row[1:] for row in matrix[1:]]
    fastaStr = ""
    
    # Extract the first column into a separate list
    ids = [row[0] for row in matrix]
    # Remove the first column from each row in the matrix
    matrix = [row[1:] for row in matrix]
    
    usedIDs = []
    
    print("started creating fasta")
    
    # Add the most common base from each row to the FASTA string
    line_length = 80
    for i in range(len(matrix)):
        most_common, count = Counter(matrix[i]).most_common(1)[0]
        if count / len(matrix[i]) <= cutoff:
            fastaStr += most_common
            usedIDs.append(ids[i])
            if len(fastaStr) % line_length == 0:
                fastaStr += "\n"
    
    print("started writing fasta")
    
    # Convert numeric bases to nucleotides
    base_map = {'0': 'A', '1': 'C', '2': 'G', '3': 'T'}
    fastaStr = ''.join([base_map.get(base, base) for base in fastaStr])
    
    with open(output_path_fasta, 'w') as fasta_file:
        fasta_file.write(">refFull\n")
        fasta_file.write(fastaStr + "\n")
    
    # Write usedIDs to a CSV file with one column
    with open(output_path_ids, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID'])
        for uid in usedIDs:
            writer.writerow([uid])
        
    
    
#create_specifier_matrix(2023, genetic_data="../data/Genetic_Data/CPB_genetic_metadata.csv", output_path="../data/Genetic_Data/specifier_matrix_2023.csv")
generate_Ref_FASTA_from_genolike(0.95)