# Looks for loops in a qmu file
import numpy as np
import time
import random
import psutil

dimensions = 10

def IndexData(fn):
    # Preprocess the data: open file and split the string right before the matrix
    file = open(fn, "r")
    data = file.read()
    data = data.split("\n//Matrix\n10 10\n")

    print("Indexing data...")

    # Convert the matrices into numpy arrays
    y = []
    for d in data[1:]:
        # Ignore the first one since it will be empty
        spl = d.split("\n")
        mat = spl[0:dimensions]
        matrix = np.zeros((dimensions, dimensions), dtype = int)
        for i in range(dimensions):
            nums = mat[i].split(" ")
            for j in range(dimensions):
                matrix[i, j] = int(nums[j])
        y.append(matrix)
    return y


# Now y is a list of around 40000-ish length containing all the exchange matrices
# It remains to run a dfs through all these ys and print the index of the ones without loops
# Run a DFS algorithm to detect loops.
def HasLoops(matrix: np.ndarray):
    # The [i,j] entry says the number of arrow from node i has an arrow to node j
    # The entries are all shifted by 1 in this implementation but we don't care
    visited_nodes = set()

    # Returns true if a loop is detected
    def DFSrecursive(matrix, node: int, ancestor: int):
        visited_nodes.add(node)
        all_edges = [i for i in range(dimensions) if matrix[node, i]  != 0]
        for vertex in all_edges:
            if vertex == ancestor: continue
            if vertex in visited_nodes: return True
            if DFSrecursive(matrix, vertex, node): return True
        return False
    
    # Run a DFS at 0 and see if there are any leftovers
    if DFSrecursive(matrix, 0, None): return True

    # Consider the leftovers
    for i in range(dimensions):
        if i in visited_nodes: continue
        if DFSrecursive(matrix, i, None): return True
    
    assert len(visited_nodes) == dimensions
    return False

# Calculate the number of arrows of a matrix
def weight(matrix):
    return int(np.sum(np.abs(matrix))/2)

def Phase1(y: list):
    t1 = time.time()
    found = False

    # The number of arrows. Loopseeker understands that whenever he is up for the job he is not likely to find any loops. Hence he also look for quivers with exceptionally low weight
    lowest_num_arrows = weight(y[0])
    minimum_weight_entry = 0

    print("There are {} entries in the file. Starting phase 1 of the search in the file...".format(len(y)))

    for i in range(len(y)):
        matrix = y[i]
        if not HasLoops(matrix):
            print("Entry {} has no loops!".format(i))
            found = True
        
        w = weight(matrix)
        if w < lowest_num_arrows:
            minimum_weight_entry = i
            lowest_num_arrows = w
            

    t2 = time.time()
    print("Time taken = %f seconds" % round(t2 - t1, 5))
    print("Lowest weight entry: {} with weight {}".format(minimum_weight_entry, lowest_num_arrows))
    return found, lowest_num_arrows

# Define mutations.
def Mutate(matrix, k: int):
    # Since we need to sort of update the whole matrix all at once we construct an identical one to keep track of stuff
    B = np.array(matrix, dtype = int)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            # Equation 3.10 in Professor Ip's lecture notes
            matrix[i, j] = -B[i, j] if (k == i or k == j) else B[i, j] + 0.5 * (np.abs(B[i, k]) * B[k , j] + B[i, k] * np.abs(B[k, j]))
    return matrix

best_found = []

# If your ram exceeds a certain percentage then it will force exit
def Phase2(original, min_weight, exit_percentage = 95):
    global best_found
    
    # Random search
    iterations = 0
    t = time.time()

    # Keep track of the current mutation sequence
    mutation_sequence = []
    checked = set()
    num_checked = 0

    last_print_len = 0

    found = False
    force_exit = False

    while not force_exit:
        B_matrix = np.array(original, dtype = int)
        local_iter = 0
        local_exit = False
        # Check if the current B_matrix exceeds the max arrow multiple
        while not force_exit and not local_exit and np.count_nonzero(np.abs(B_matrix) > 50) <= 3 and local_iter < 1e5:
            # Decide on a spot to mutate
            random_number = random.randint(0, dimensions - 1)

            # If it is the last mutation then might as well not do it... try mutate at a different spot
            while len(mutation_sequence) > 0 and random_number + 1 == mutation_sequence[-1]:
                random_number = random.randint(0, dimensions - 1)
            
            # Do the mutation
            B_matrix = Mutate(B_matrix, random_number)
            mutation_sequence.append(random_number + 1)

            if tuple(mutation_sequence) in checked:
                continue

            w = weight(B_matrix)

            # Check the weight
            if not found and (w < min_weight or (w == min_weight and len(mutation_sequence) < len(best_found))):
                print("\nThis is a low {} weight entry".format(w))
                print(mutation_sequence)
                # Update lowest and overwrite print eraser
                best_found = mutation_sequence
                min_weight = w
                last_print_len = 0
                local_exit = True

            # Check for loops. No need to run DFS if it is already checked
            # Use a dumber algorithm than dfs to check first. It is a necessary condition for an entry with no loops to have #arrows = #vertices - 1
            # However the condition is not sufficient since the graph might just be disjoint
            # I believe the indexing is done using something like BST so it will be a bit faster this way
            if not found and w <= dimensions - 1 and not HasLoops(B_matrix):
                print("\nThis entry has no loops!!!!")
                print(mutation_sequence)
                # Mark found but also overwrite the print eraser
                found = True
                last_print_len = 0
                best_found = mutation_sequence
                local_exit = True

            # See if we can find a shorter one if we already found something. Guarding it with the found condition so it exits early if we havent even found anything
            if found and not HasLoops(B_matrix) and len(mutation_sequence) < len(best_found):
                print("\nThis entry has no loops and it is a shorter sequence than the previous one. It has length {}".format(len(mutation_sequence)))
                print(mutation_sequence)
                last_print_len = 0
                best_found = mutation_sequence
                local_exit = True
            
            # Alright this must be good enough. Exiting
            if found and len(best_found) < dimensions:
                print("Alright this is good enough, exiting...")
                last_print_len = 0
                force_exit = True

            checked.add(tuple(mutation_sequence))
            local_iter += 1

        # Reset stuff
        iterations += 1
        mutation_sequence = []

        # If it exit due to exploding local iteration, then finite type then the quiver will not explode and there will be too many local iterations
        if local_iter == 1e5:
            print("\nThe quiver is not exploding. It probably has finite type or finite mutation type.")
            force_exit = True

        # Print things every 16 iterations
        if iterations & 15 == 0 and not force_exit:
            # Print result
            t0 = time.time()
            if not found: print_str = "Current iterations = {}, checked instances = {}, lowest weight = {}, time taken so far = {} minutes.".format(iterations, len(checked) + num_checked, min_weight, round((t0 - t)/60, 5))
            else: print_str = "Current iterations = {}, checked instances = {}, shortest sequence length = {}, time taken so far = {} minutes.".format(iterations, len(checked) + num_checked, len(best_found), round((t0 - t)/60, 5))
            print("\b" * last_print_len + print_str, end = "", flush = True)
            last_print_len = len(print_str)
            
            # Reset when ram is exploding... usually we run these overnight
            if psutil.virtual_memory().percent > exit_percentage:
                force_exit = True
                del checked
                time.sleep(1)
    
    print()
    print("Stopping phase 2...")
    return found


if __name__ == "__main__":
    fn = input("Enter filename:")
    y = IndexData(fn)
    found, min_weight = Phase1(y)
    if not found:
        print("Nothing found, starting phase 2 of the search...")
    found = Phase2(y[0], min_weight)
    if not found:
        print("Nothing found :(")