import numpy as np
import loopseeker as ls
import tensordiagram as td

b = "black"
w = "white"

file_pream = "./grassmannian_results"

def DetermineBestSequence(i, signature, case_name):
    # Generate the diagram and get the exchange matrix
    diagram = td.TriangulationDiagram(signature)
    diagram.GenerateCluster()
    matrix = np.array(diagram.cluster.exchange_matrix, dtype = int)

    # Find the best mutation sequence for the loop
    ls.dimensions = matrix.shape[0]
    ls.Phase2(matrix, ls.weight(matrix), exit_percentage = 90)

    # Write the result to the file
    with open("{}/result.txt".format(file_pream), "a") as f:
        write_str = "The case {} with signature {}, best mutation sequence = {}\n".format(i, str(signature), str(ls.best_found))
        f.write(write_str)
        f.seek(0)

    # Generate the qmu file for result verification
    fn = "{}/original_{}{}.qmu".format(file_pream, case_name, i)
    with open(fn, "a") as f:
        st = td.GenerateQMUfile(matrix, fn.format(case_name, i))
        f.write(st)
    
    # Maybe let's also save a copy of the mutated one so we can do our job quicker
    for k in ls.best_found:
        matrix = ls.Mutate(matrix, k - 1)

    fn2 = "{}/mutated_{}{}.qmu".format(file_pream, case_name, i)
    with open(fn2, "a") as f:
        st = td.GenerateQMUfile(matrix, fn2.format(case_name, i))
        f.write(st)
    return


if __name__ == "__main__":
    # Flush previous results to avoid confusion
    with open("{}/result.txt".format(file_pream), "a") as f:
        write_str = "\n" * 300
        f.write(write_str)
        f.seek(0)
    
    # Perform the search
    for i in range(5, 7):
        print("Considering the case n = {}".format(i))
        DetermineBestSequence(i, [b for _ in range(i-3)] + [w, w, w], "expt")
        DetermineBestSequence(i, [b for _ in range(i)], "grsm")