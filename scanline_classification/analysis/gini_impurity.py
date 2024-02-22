import subprocess
from itertools import product

curvature_threshold = [20, 30, 40, 50, 60]
std_multiplier = [1, 100, 200, 300]
neighborhood_multiplier = [1, 2, 3, 4]

# Generate all combinations
combinations = list(product(curvature_threshold, std_multiplier, neighborhood_multiplier))

# Print the combinations
for i, (cv_t, std_m, neigh_m) in enumerate(combinations):
    print("---------------------------------------------------")
    print(f'Combination --- Curvature threshold:{cv_t} | Std multiplier:{std_m} | Neighborhood multiplier:{neigh_m} --- {i+1}/{len(combinations)}')
    print("---------------------------------------------------")
                
    command = (
    f"python scanline_classification/scanline_classification_main.py "
    f"scs.curvature_threshold={cv_t} "
    f"scs.std_multiplier={std_m} "
    f"scs.neighborhood_multiplier={neigh_m}"
    )
    
    # Run the command
    subprocess.run(command, shell=True)

    # # Run the command
    # process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)

    # # Wait for the command to finish
    # process.wait()

    # # Print the output
    # print(process.stdout.read())


