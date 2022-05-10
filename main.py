from re import L
import numpy as np
from tensordiagram import *
from loopseeker import search
import time

def save_result(st):
    result_fn = "./result.txt"
    with open(result_fn, "a") as f:
        f.write(st + "\n")
        
b = "black"
w = "white"
qmu_path = "./quiver/batch2/"

for n in range(7, 15):
    print("Searching n = ", n)

    # Grassmannian
    gr = TriangulationDiagram([b for _ in range(n)])
    gr.GenerateCluster()

    # PGrassmannian
    pgr = TriangulationDiagram([w, w, w] + [b for _ in range(n - 3)] )
    pgr.GenerateCluster()

    # Create a copy, just to be safe
    B_gr = np.array(gr.cluster.exchange_matrix, dtype = int)
    B_pgr = np.array(pgr.cluster.exchange_matrix, dtype = int)

    # Save both qmu files
    GetQMU(B_gr, qmu_path + "gm" + str(n), save_file = True)
    GetQMU(B_pgr, qmu_path + "pgm" + str(n), save_file = True)

for n in range(7, 12):
    print("Searching n = ", n)

    # Grassmannian
    gr = TriangulationDiagram([b for _ in range(n)])
    gr.GenerateCluster()

    # PGrassmannian
    pgr = TriangulationDiagram([w, w, w] + [b for _ in range(n - 3)] )
    pgr.GenerateCluster()

    # Create a copy, just to be safe
    B_gr = np.array(gr.cluster.exchange_matrix, dtype = int)
    B_pgr = np.array(pgr.cluster.exchange_matrix, dtype = int)

    # Find the shortest path
    mut_seq, relabelling = search(B_pgr, B_gr, 8)
    B_mut_pgr = Mutate(B_pgr, mut_seq)
    B_mut_pgr = B_mut_pgr[relabelling]
    B_mut_pgr = B_mut_pgr[:, relabelling]
    assert np.count_nonzero(B_gr == B_mut_pgr) == (2 * n - 8) ** 2
    result_text = f"For n = {n}, shortest mutation sequence = {mut_seq}"
    save_result(result_text)
    print(result_text)