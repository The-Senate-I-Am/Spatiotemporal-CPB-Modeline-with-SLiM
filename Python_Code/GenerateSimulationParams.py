from pathlib import Path
import numpy as np

#This function determines migration rates based on distances between clusters given into distances as a numpy array.
#total_migration is HOW MUCH: the immigrant fraction per subpop per generation (must stay < 1,
#SLiM takes retention as 1 - total_migration). scale is only WHERE FROM: the decay of the
#dispersal kernel e^(-x*scale). Each row's off-diagonals sum to exactly total_migration.
#The output is a numpy array of migration rates, and the function also saves them to a CSV file.
def determine_migration_rates(distances, total_migration=0.05, scale=0.0001, output_path=Path('../data/migration_rates.csv')):
    migration_rates = np.copy(distances).astype(float)  # Create a copy of distances to store migration rates
    for i in range(1, len(distances)):
        # Dispersal kernel over SOURCE subpops only (exclude self); sets where migrants come from.
        weights = np.array([np.exp(-distances[i][j] * scale) if j != i else 0.0
                            for j in range(1, len(distances[i]))])
        kernel = weights / weights.sum()  # normalize sources to 1 (conditional on migrating)

        # Scale the kernel by the total immigration fraction; off-diagonals now sum to total_migration.
        for idx, j in enumerate(range(1, len(distances[i]))):
            migration_rates[i][j] = total_migration * kernel[idx]

    # Save the migration rates to a CSV file
    np.savetxt(output_path, migration_rates, delimiter=",", fmt='%.6f')
    return migration_rates

