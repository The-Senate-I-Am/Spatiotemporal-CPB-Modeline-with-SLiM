from pathlib import Path
import numpy as np

#This function determines migration rates based on distances between clusters given into distances as a numpy array.
#Migration is decoupled into two independent quantities:
#  - total_migration: the total fraction of a subpop's offspring that are immigrants each generation
#    (a real, bounded per-generation migration rate). SLiM sets the self/retention fraction to
#    1 - total_migration automatically, so this must stay below 1.
#  - scale: the distance-decay of the dispersal KERNEL, i.e. WHERE migrants come from given that
#    migration happens (y = e^(-x*scale)). It reshapes the sources but no longer controls the amount.
#The kernel is computed over source subpops only (self excluded), normalized to sum to 1, then scaled
#by total_migration so each row's off-diagonals sum to exactly total_migration.
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

