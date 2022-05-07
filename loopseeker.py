# Looks for loops in a qmu file
import numpy as np
import time
import random
import psutil
from tensordiagram import weight, Mutate
from utility import *

dimensions = 10

# Retrieve the mutation class obtained from qmu
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

# Phase 1: search through a list of mutated quivers to see if they have loops. Typically they are generated from the qmu file
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

# Takes in the quiver and the target quiver, returns a list which is the mutation sequence and the relabelling
def search(B_matrix, target, max_search_length, try_first = []):
    # initiate a empty list to keep track of the current mutation sequence
    # Try all the "try first" sequences
    # while queue is not empty: (which it will practically never be unless we have reached max search length)
    #   take a sequence s from queue
    #   do the mutation according to q
    #   For every node at quiver:
    #       mutate at node
    #       if the quiver is equivalent to target up to relabelling:
    #           return this sequence
    #   else add all mutation sequences to this quiver's neighbour into the queue
    num_nodes = B_matrix.shape[0]
    assert  B_matrix.shape[0] ==  B_matrix.shape[1]
    if len(try_first) > 0:
        for seq in try_first:
            B = Mutate(B_matrix, seq)
            is_relabel, relabelling = is_quiver_relabelling(B, target)
            if is_relabel:
                return seq, relabelling
        print("The try first ones are not right! Doing regular search...")
    q = [[]]
    while len(q) > 0:
        seq = q.pop(0)
        B = Mutate(B_matrix, seq)
        for i in range(num_nodes):
            B1 = Mutate(B, i)
            is_relabel, relabelling = is_quiver_relabelling(B1, target)
            if is_relabel:
                seq.append(i)
                return seq, relabelling

        if len(seq) < max_search_length:
            for i in range(num_nodes):
                q.append(seq + [i])

# Given two quiver see if they are equivalent up to relabelling
def is_quiver_relabelling(B_matrix, target):
    # I am told this is an NP problem so let's first figure out when it is definitely not isomorphic
    # If there is a different number of arrows or nodes it is definitely not equivalent up to relabelling
    # If it is the exact same then its definitely equivelent up to relabeling
    if B_matrix.shape != target.shape: return False, np.array([])
    if weight(B_matrix) != weight(target): return False, np.array([])
    num_nodes = B_matrix.shape[0]
    if np.count_nonzero(B_matrix == target) == num_nodes ** 2: return True, np.arange(num_nodes)
    # If they do not have the same amount of each multiple arrows it is definitely not equivalent up to relabelling
    # B_pos = np.abs(B_matrix)
    # T_pos = np.abs(target)
    B_arrow_weights, B_arrow_counts = np.unique(B_matrix, return_counts = True)
    T_arrow_weights, T_arrow_counts = np.unique(target, return_counts = True)
    if B_arrow_weights.shape != T_arrow_weights.shape: return False, np.array([])
    if np.count_nonzero(B_arrow_weights == T_arrow_weights) < B_arrow_weights.shape[0]: return False, np.array([])
    if np.count_nonzero(B_arrow_counts == T_arrow_counts) < B_arrow_counts.shape[0]: return False, np.array([])
    # If the number of nodes with a given valence(# edges incident to it) cannot be paired up one by one, it is definitely not equivalent up to relabelling
    # We also create a dictionary of valence so that we can check drastically smaller amount of cases later
    # Keys in the valence dict is a tuple of (#incoming arrows, #outgoing arrows), values is a list of node index with said valence
    # print("We are at the relabeling part. Have fun.")
    B_valence, T_valence = {}, {}

    def get_valence(valence, matrix, i):
        row_i = np.array(matrix[i], dtype = int)
        num_in_arrows = -np.sum(row_i[row_i < 0])
        num_out_arrows = np.sum(row_i[row_i > 0])
        key = (num_in_arrows, num_out_arrows)
        if key in valence.keys():
            valence[key].append(i)
        else:
            valence[key] = [i]
        
    for i in range(num_nodes):
        get_valence(B_valence, B_matrix, i)
        get_valence(T_valence, target, i)
    
    # print(sorted(B_valence.keys()), sorted(T_valence.keys()))
    if sorted(B_valence.keys()) != sorted(T_valence.keys()): return False, np.array([])
    # We feel like we ran out of ideas so we just try every reindexing now.
    # Make an empty list Q to store all the possible "sensible" reindexings in the sense of matching valence
    # Also keep track of the number n of permutations generated up to the last key
    # for key in B valence dictionary:
    #   for i < n:
    #       pop the first element s from Q which should be a numpy array
    #       generate permutations of B_valence[key] which is a list of node indices
    #       for each permutation:
    #           copy s to a new numpy array
    #           using array indexing magic, we can put the permutation in place by s[T_valence[key]] = permutation
    #           append this numpy array to Q
    #   update num_sequences
    queue = [np.zeros((num_nodes,), dtype = int)]
    num_sequences = 1
    for key in B_valence.keys():
        for _ in range(num_sequences):
            seq = queue.pop(0)
            perm = np.array(B_valence[key], dtype = int)
            # Generate using Heap algorithm
            n = len(perm)
            stack_state = np.zeros((n, ), dtype = int)
            # First output the original
            seq_copy = np.array(seq, dtype = int)
            seq_copy[T_valence[key]] = perm
            queue.append(seq_copy)
            n = len(perm)
            i = 0
            while i < n:
                if stack_state[i] < i:
                    if i % 2 == 0:
                        perm[[0, i]] = perm[[i, 0]]
                    else:
                        perm[[stack_state[i], i]] = perm[[i, stack_state[i]]]
                    seq_copy = np.array(seq, dtype = int)
                    seq_copy[T_valence[key]] = perm
                    queue.append(seq_copy)
                    stack_state[i] += 1
                    i = 0
                else:
                    stack_state[i] = 0
                    i += 1
        num_sequences = len(queue)
    
    # Now try every permutation
    for i in range(num_sequences):
        perm = queue[i]
        B = np.array(B_matrix, dtype = int)
        B = B[:, perm]
        B = B[perm]
        if np.count_nonzero(B == target) == num_nodes ** 2:
            return True, perm
        fprint(i, num_sequences, "Trying permutation ")
    return False, np.array([])

if __name__ == "__main__":
    fn = input("Enter filename:")
    y = IndexData(fn)
    found, min_weight = Phase1(y)
    if not found:
        print("Nothing found, starting phase 2 of the search...")
    found = Phase2(y[0], min_weight)
    if not found:
        print("Nothing found :(")