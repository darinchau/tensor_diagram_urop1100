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
best_B = np.zeros((1, 1))

# If your ram exceeds a certain percentage or if the number of checked instances exceeds the force_ex threshold then it will force exit
# Find shorter: if set to True, the loop will continue to run even if we find an entry with no loops
# Try find shortest: if set to True, the program will always try to find the shortest mutation sequence from the original
def Phase2(original, min_weight, exit_percentage = 95, find_shorter = True, force_ex: int = 10000, try_find_shortest = True):
    global best_found
    global best_B
    best_B = np.zeros((dimensions, dimensions), dtype = int)
    
    # Random search
    iterations = 0
    t = time.time()

    # Keep track of the current mutation sequence
    mutation_sequence = []
    checked = set()
    num_checked = 0

    last_print_len = 0

    found, force_exit, local_exit = False, False, False
    B_matrix = np.array(original, dtype = int)

    # Main loop
    while not force_exit:

        local_exit = False
        mutation_sequence = []

        # If all we want is to find a good sequence then why not start from the best sequence so far
        if not try_find_shortest:
            for m in best_found:
                mutation_sequence.append(m)

        # Check if the current B_matrix exceeds the max arrow multiple
        while (not force_exit) and (not local_exit):
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
            # Test 1: If we have not found a quiver with no loops then we update the best sequence
            if not found and (w < min_weight or (w == min_weight and len(mutation_sequence) < len(best_found))):
                print("\nThis is a low {} weight entry".format(w))
                print(mutation_sequence)
                # Update lowest and overwrite print eraser
                if not try_find_shortest:
                    best_B = np.array(B_matrix)
                best_found = mutation_sequence
                min_weight = w
                last_print_len = 0
                local_exit = True

            # Check for loops. No need to run DFS if it is already checked
            # Use a dumber algorithm than dfs to check first. It is a necessary condition for an entry with no loops to have #arrows = #vertices - 1
            # However the condition is not sufficient since the graph might just be disjoint
            # I believe the indexing is done using something like BST so it will be a bit faster this way

            # Test 2:  If we have not found a quiver with no loops but now we found one, then we proceed
            if not found and w <= dimensions - 1 and not HasLoops(B_matrix):
                print("\nThis entry has no loops!!!!")
                print(mutation_sequence)
                found = True
                if not try_find_shortest:
                    best_B = np.array(B_matrix)
                last_print_len = 0
                best_found = mutation_sequence
                local_exit = True
                # If the instruction says we do not need to find a shorter sequence, then we bounce
                if not find_shorter: force_exit = True

            # See if we can find a shorter one if we already found something. Guarding it with the found condition so it exits early if we havent even found anything
            # Test 3: if we have already found a sequence with no loops but we found a shorter sequence, then we update
            if found and not HasLoops(B_matrix) and len(mutation_sequence) < len(best_found):
                print("\nThis entry has no loops and it is a shorter sequence than the previous one. It has length {}".format(len(mutation_sequence)))
                print(mutation_sequence)
                if not try_find_shortest:
                    best_B = np.array(B_matrix)
                last_print_len = 0
                best_found = mutation_sequence
                local_exit = True
            
            # If we found a sufficiently short sequence, we force exit
            # Alright this must be good enough. Exiting
            if found and len(best_found) < dimensions:
                print("Alright this is good enough, exiting...")
                last_print_len = 0
                force_exit = True
            
            # Check exit conditions            
            if len(mutation_sequence) > 40000:
                print("The quiver is not exploding! It is probably a loop finite mutation type")
                force_exit = True
            
            if np.count_nonzero(np.abs(B_matrix) > 50) > 3:
                local_exit = True

            checked.add(tuple(mutation_sequence))

        # Reset stuff
        iterations += 1
        B_matrix = np.array(original, dtype = int) if try_find_shortest else np.array(best_B, dtype = int)

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

            if iterations > force_ex:
                force_exit = True
        
        if force_exit:
            print()
            print("Force Exiting...")
    
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