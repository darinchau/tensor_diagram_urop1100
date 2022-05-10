from re import L
import numpy as np
from tensordiagram import *
from loopseeker import search
import time

def save_result(st):
    result_fn = "./result.txt"
    with open(result_fn, "a") as f:
        f.write(st + "\n")

def nice_triangulation(n):
    return [(0,2), (0, n-2), (2, n-2)] + Triangulation.GenerateTriangulation(list(range(2, n-1)))
        
b = "black"
w = "white"
nam = "gr"
qmu_path = "./quiver/batch2/"

for n in range(7, 12):
    # make a signal cuz this takes long
    print("Searching n = ", n)

    short_sig = [b for _ in range(n - 3)]

    # Grassmannian
    gr = TriangulationDiagram([b, b] + [b for _ in range(n - 3)] + [b])
    gr.GenerateCluster()

    # PGrassmannian
    pgr = TriangulationDiagram([w, w] + [b for _ in range(n - 3)] + [w])
    pgr.GenerateCluster()

    # Create a copy, just to be safe
    B_gr = np.array(gr.cluster.exchange_matrix, dtype = int)
    B_pgr = np.array(pgr.cluster.exchange_matrix, dtype = int)

    # Save both qmu files
    GetQMU(B_gr, qmu_path + "black" + nam + str(n), save_file = True)
    GetQMU(B_pgr, qmu_path + "white" + str(n), save_file = True)

    # Find the shortest path
    mut_seq, relabelling = search(B_pgr, B_gr, 8)
    B_mut_pgr = Mutate(B_pgr, mut_seq)
    B_mut_pgr = B_mut_pgr[relabelling]
    B_mut_pgr = B_mut_pgr[:, relabelling]
    assert np.count_nonzero(B_gr == B_mut_pgr) == (2 * n - 8) ** 2
    result_text = f"For n = {n}, name = {nam}, shortest mutation sequence = {mut_seq}"
    save_result(result_text)
    print(result_text)