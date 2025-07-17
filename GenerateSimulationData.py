import numpy as np

#This function determines migration rates based on distances between clusters given into distances as a numpy array.
#The distribution used is the exponential distribution (y = e^(-x/m)) where m is the modifier, x is the distance, and y is the unadjusted migration rate.
#The function is calculated for each distance in the a row on the distance matrix, including for the case of zero distance. (distance to self).
#These rates are then added up and normalized to sum to 1.
#The output is a numpy array of migration rates, and the function also saves the migration rates to a CSV file. Specify 'none' for no saving to csv
def determine_migration_rates(distances, modifier=10000, output_path='./data/migration_rates.csv'):
    migration_rates = np.zeros((len(distances), len(distances[0])))
    for i in range(1, len(distances)):
        total_rate = 0
        
        for j in range(1, distances[i]):
            migration_rates[i][j] = np.exp(-distances[i][j] / modifier)
            total_rate += migration_rates[i][j]
            
        # Normalize the migration rates for the row
        migration_rates[i] /= total_rate
        
    # Save the migration rates to a CSV file
    if output_path.lower() != 'none':
        np.savetxt(output_path, migration_rates, delimiter=",", fmt='%.6f')
    return migration_rates

