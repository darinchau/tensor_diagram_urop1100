from re import L
import numpy as np
from tensordiagram import *
from loopseeker import search
import time

def save_result(st):
    result_fn = "./result.txt"
    with open(result_fn, "a") as f:
        f.write(st + "\n")

if __name__ == "__main__":
    for n in range(12, 15):
        print("Searching n = ", n)
        gr = TriangulationDiagram([b for _ in range(n)])
        pgr = TriangulationDiagram([b for _ in range(n - 3)] + [w, w, w])
        gr.GenerateCluster()
        pgr.GenerateCluster()
        # Create a copy, just to be safe
        B_gr = np.array(gr.cluster.exchange_matrix, dtype = int)
        B_pgr = np.array(pgr.cluster.exchange_matrix, dtype = int)
        mut_seq, relabelling = search(B_pgr, B_gr, 8, try_first=[[4, 2, 1, 5, 7, 8]])
        B_mut_pgr = Mutate(B_pgr, mut_seq)
        B_mut_pgr = B_mut_pgr[relabelling]
        B_mut_pgr = B_mut_pgr[:, relabelling]
        assert np.count_nonzero(B_gr == B_mut_pgr) == (2 * n - 8) ** 2
        result_text = f"For n = {n}, shortest mutation sequence = {mut_seq}"
        save_result(result_text)
        print(result_text)