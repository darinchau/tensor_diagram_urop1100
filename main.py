from re import L
import numpy as np
from tensordiagram import *
from loopseeker import search
import time
import random
from numpy.linalg import inv

b = "black"
w = "white"
nam = "gr"
qmu_path = "./quiver/batch5/"

# Clear result
with open("./result.txt", "w") as f:
    f.write("")

def save_result(st = "", end = "\n"):
    if type(st) != str: st = str(st)
    print(st, end = end)
    result_fn = "./result.txt"
    with open(result_fn, "a") as f:
        f.write(st + end)

def nice_triangulation(n):
    diags = [(0,2), (0, n-2), (2, n-2)] + Triangulation.GenerateTriangulation(list(range(2, n-1)))
    return Triangulation(n, diags)

def GenerateRandomSignature(n):
    return [b if random.random() < 0.5 else w for _ in range(n)]

def GetName(sig):
    return "".join(["b" if s == b else "w" for s in sig])

def GenCluster(signature):
    save_result(GetName(signature))
    a = TriangulationDiagram(signature)
    # a.Triangulate(nice_triangulation(len(signature)))
    a.GenerateCluster()
    save_result(" Cluster: ", end = " ")
    for cv in a.clusterVariables:
        save_result(cv, end = " ")
    save_result(" Frozen: ", end = " ")
    for fv in a.frozenVariables:
        save_result(fv, end = " ")
    save_result()
    return a

def TryShortSignature(short_sig, nam, n, marked_points):
    # make a signal cuz this takes long
    save_result(f"Searching n = {n}")

    try:
        gr = GenCluster(Permute([marked_points[0], marked_points[1], b, b, b, marked_points[2]] + short_sig, 3))
    except AssertionError as err:
        save_result(err)
        return
    
    try:       
        # PGrassmannian
        pgr = GenCluster(Permute([marked_points[0], marked_points[1], w, w, w, marked_points[2]] + short_sig, 3))
    except AssertionError as err:
        save_result(err)
        return

    # Create a copy, just to be safe
    B_gr = np.array(gr.cluster.exchange_matrix, dtype = int)
    B_pgr = np.array(pgr.cluster.exchange_matrix, dtype = int)

    # Find the shortest path
    save_result("Searching...")
    mut_seq, relabelling = search(B_pgr, B_gr, 8)
    B_mut_pgr = Mutate(B_pgr, mut_seq)
    B_mut_pgr = B_mut_pgr[relabelling]
    B_mut_pgr = B_mut_pgr[:, relabelling]

    save_result()
    # Save QMU
    if np.count_nonzero(B_gr != B_mut_pgr) == 0:
        GetQMU(gr.cluster.full_exchange_matrix, qmu_path + GetName(marked_points) + "_" + nam + "_black" + str(n), n, save_file = True)
        GetQMU(pgr.cluster.full_exchange_matrix, qmu_path + GetName(marked_points) + "_" + nam + "_white" + str(n), n, save_file = True)
        for i in range(4):
            save_result(gr.cluster.exchangeRelations[i])
            save_result(pgr.cluster.exchangeRelations[i]) 
            save_result()     
                 
    elif np.count_nonzero(B_gr != -B_mut_pgr) == 0:
        GetQMU(gr.cluster.full_exchange_matrix, qmu_path + GetName(marked_points) + "_" + nam + "_black" + str(n), n, save_file = True)
        GetQMU(-pgr.cluster.full_exchange_matrix, qmu_path + GetName(marked_points) + "_" + nam + "_white" + str(n), n, save_file = True)
        for i in range(4):
            save_result(gr.cluster.exchangeRelations[i])       
            pgr.cluster.exchangeRelations[i].swap()
            save_result(pgr.cluster.exchangeRelations[i]) 
            save_result() 
    else:
        raise AssertionError       

    result_text = f"For n = {n}, name = {nam}, shortest mutation sequence = {mut_seq}"
    save_result(result_text)
    save_result("\n\n\n")
    return

def test1():
    tried = []
    for mp in [[b,b,b], [b,b,w], [b,w,b], [b,w,w], [w,b,b], [w,b,w], [w,w,b], [w,w,w]]:
        save_result(f"Trying {mp}")
        while len(tried) < 8:
            n = 9
            short_sig = GenerateRandomSignature(n-6)
            nam = GetName(short_sig)
            if (n, nam) in tried: continue
            tried += [(n, nam)]
            save_result("Trying the case for " + nam)
            TryShortSignature(short_sig, nam, n, mp)
        save_result("=" * 99)
        tried = []

def test2():
    test_sig = [b, b]
    black = [test_sig[0]] + [b, b, b] + [test_sig[1]]
    white = [test_sig[0]] + [w, w, w] + [test_sig[1]]
    for _ in range(10):
        n = 9
        
        s = GenerateRandomSignature(n-5)
        st = "*" * 60 + f" Signature: {GetName(s)} " + "*" * 60
        
        print("*" * len(st) + "\n" + st + "\n" + "*" * len(st))
        print("=" * 100 + " black " + "=" * 100)
        GetInfo(TriangulationDiagram(Permute(black + s, -2)), printer = True)

        print("=" * 100 + " white " + "=" * 100)
        GetInfo(TriangulationDiagram(Permute(white + s, -2)), printer = True)
        print("\n"*10)

def test3():
    p = 0
    q = 5
    r = 7
    pqr_col = [w, w, b]
    sig1 = [pqr_col[0], invert(pqr_col[0]), pqr_col[0], pqr_col[0], pqr_col[1], invert(pqr_col[1]), invert(pqr_col[1]), pqr_col[2], invert(pqr_col[2]), pqr_col[2], pqr_col[2]]
    sig2 = [pqr_col[0], pqr_col[0], pqr_col[0], pqr_col[0], pqr_col[1], invert(pqr_col[1]), invert(pqr_col[1]), pqr_col[2], invert(pqr_col[2]), pqr_col[2], pqr_col[2]]

def test4():
    sig = [w, b, w, b, w, b]
    a = TensorDiagram(sig)
    a.AddJInvariant(a.J1(0, 3))
    a.localscale = 0.5
    a.dotsize = 6
    a.scale = 1
    save_result(a)

def test5():
    sig = [b, w, b,b, w,w, b,b,b, w,w,w]
    print(sig)
    print(Permute(sig, 2))
    print(Permute(sig, 5))
    print(Permute(sig, 0))
    print(Permute(sig, -4))

def test6():
    ss = [w, b, b]
    sigb = [b,b,b] + ss + [b,b,b]
    sigw = [w,w,b] + ss + [b,b,w]
    ab = TensorDiagram(sigb)
    aw = TensorDiagram(sigw)
    aw.AddJInvariant(aw.J1(1, 2))
    ab.AddJInvariant(ab.J1(1, 0))
    save_result(aw)
    save_result(ab)

def test7():
    sig = [w,w,b,b,w,w,b,b,w]
    a = TriangulationDiagram(sig)
    c = TriangulationDiagram(sig)
    a.AddJInvariant(a.J4(3,8,0,2))
    c.AddJInvariant(c.J1(0,3))
    save_result(a)
    save_result(c)
    

def TryShortSignature2(short_sig, nam, n, marked_points):
    # make a signal cuz this takes long
    save_result(f"Searching n = {n}")

    try:
        gr = GenCluster(Permute([marked_points[0], marked_points[1], b, b, b, marked_points[2]] + short_sig, 3))
    except AssertionError as err:
        save_result(err)
        return
    
    try:       
        # PGrassmannian
        pgr = GenCluster(Permute([marked_points[0], marked_points[1], w, w, w, marked_points[2]] + short_sig, 3))
    except AssertionError as err:
        save_result(err)
        return

    # Create a copy, just to be safe
    B_gr = np.array(gr.cluster.exchange_matrix, dtype = int)
    B_pgr = np.array(pgr.cluster.exchange_matrix, dtype = int)

    # Find the shortest path
    save_result("Searching...")
    mut_seq, relabelling = search(B_pgr, B_gr, 8)
    B_mut_pgr = Mutate(B_pgr, mut_seq)
    B_mut_pgr = B_mut_pgr[relabelling]
    B_mut_pgr = B_mut_pgr[:, relabelling]

    save_result()
    # Save QMU
    B_gr_full = np.array(gr.cluster.full_exchange_matrix, dtype = int)
    B_pgr_full = np.array(pgr.cluster.full_exchange_matrix, dtype = int)
    B_pgr_full = Mutate(B_pgr_full, mut_seq)
    if np.count_nonzero(B_gr != B_mut_pgr) == 0:
        pass        
    elif np.count_nonzero(B_gr != -B_mut_pgr) == 0:
        B_pgr_full *= -1
    else:
        raise AssertionError  

    B_tilda = B_gr_full[10:15:, :5]
    B_tilda_bar = B_pgr_full[10:15, :5]
    print(B_tilda.shape, B_tilda_bar.shape)
    M_psi = np.dot(inv(B_tilda), B_tilda_bar)
    M_psi_inv = inv(M_psi)
    save_result(M_psi)
    return

def test8():
    tried = []
    for mp in [[b,b,b], [b,b,w], [b,w,b], [b,w,w], [w,b,b], [w,b,w], [w,w,b], [w,w,w]]:
        save_result(f"Trying {mp}")
        while len(tried) < 8:
            n = 9
            short_sig = GenerateRandomSignature(n-6)
            nam = GetName(short_sig)
            if (n, nam) in tried: continue
            tried += [(n, nam)]
            save_result("Trying the case for " + nam)
            TryShortSignature2(short_sig, nam, n, mp)
        save_result("=" * 99)
        tried = []

def test9():
    a = TriangulationDiagram([b,b,b,b,b])
    a.PlotTriangulation((-1.75, 0), 1)

    a.Triangulate(Triangulation(5, [(4,1), (4,2)]))
    a.PlotTriangulation((1.75, 0), 1)

def test10():
    a = TensorDiagram([b,b,b,w,b,b,b,w])
    a.AddJInvariant(a.J1(2, 6))
    a.dotsize = 1.5
    save_result(a)

def test11():
    pgr = TriangulationDiagram([w,w,b,b,b,b,b,b,w])
    gr = TriangulationDiagram([b] * 9)
    pgr.GenerateCluster()
    gr.GenerateCluster()

    GetQMU(pgr.cluster.full_exchange_matrix, "./quiver/batch5/pgr9", 9, True)
    GetQMU(gr.cluster.full_exchange_matrix, "./quiver/batch5/gr9", 9, True)