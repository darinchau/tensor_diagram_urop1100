import numpy as np
import loopseeker as ls
import tensordiagram as td
import time

b = "black"
w = "white"
result_fn = ""

file_pream = "./gr_results_batch_2"

def DetermineBestSequence(i, signature, case_name):
    # Generate the diagram and get the exchange matrix
    diagram = td.TriangulationDiagram(signature)
    diagram.GenerateCluster()
    matrix = np.array(diagram.cluster.exchange_matrix, dtype = int)

    # Find the best mutation sequence for the loop
    ls.dimensions = matrix.shape[0]
    ls.Phase2(matrix, ls.weight(matrix), exit_percentage = 90, find_shorter = False, force_ex = 100000, try_find_shortest = False)

    # Write the result to the file
    with open(result_fn, "a") as f:
        write_str = "The case {} with signature {}, best mutation sequence of {} length = {}\n".format(i, str(signature), len(ls.best_found) ,str(ls.best_found))
        f.write(write_str)
        f.seek(0)

    # Generate the qmu file for result verification
    fn = "{}/original_{}{}.qmu".format(file_pream, case_name, i)
    with open(fn, "w") as f:
        st = td.GenerateQMUfile(matrix, fn.format(case_name, i))
        f.write(st)
    
    # Maybe let's also save a copy of the mutated one so we can do our job quicker
    for k in ls.best_found:
        matrix = ls.Mutate(matrix, k - 1)

    fn2 = "{}/mutated_{}{}.qmu".format(file_pream, case_name, i)
    with open(fn2, "w") as f:
        st = td.GenerateQMUfile(matrix, fn2.format(case_name, i))
        f.write(st)
    return


if __name__ == "__main__":
    # Create the file
    result_fn = "{}/result.txt".format(file_pream)
    with open(result_fn, "w") as f:
        f.write("")
    
    # Perform the search
    t = time.time()
    for i in range(10, 15):
        print("Considering the case n = {}".format(i))
        DetermineBestSequence(i, [b for _ in range(i-3)] + [w, w, w], "expt")
        DetermineBestSequence(i, [b for _ in range(i)], "grsm")
    t2 = time.time()
    print("Time taken = {}".format(round((t2 - t)/60 , 5)))