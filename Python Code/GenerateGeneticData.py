import pandas as pd

def create_specifier_matrix(year, genetic_data="../data/Genetic_Data/CPB_genetic_metadata.csv", output_path="../data/Genetic_Data/specifier_matrix.csv"):
    """
    Create a specifier matrix for the given year from genetic data. This matrix will have 
    
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

    
create_specifier_matrix(2023, genetic_data="../data/Genetic_Data/CPB_genetic_metadata.csv", output_path="../data/Genetic_Data/specifier_matrix_2023.csv")