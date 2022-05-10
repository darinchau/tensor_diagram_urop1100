import numpy as np
import time

## CONSTANTS ##
PI = 3.14159265
EPSILON = 1e-5

# Graph plotting
vertexThreshold = 0.1
vertexRepellance = 0.3
edgeThreshold = 0.2
edgeRepellance = 0.2

# Number of circles we offset for plotting. 0 means node 0 is at right hand side; 1/4 means we start at the top etc
plotOffsetAngle = 0.5

# Technically this thing is subtraction free so two symbols suffices
plus = "+"
equal = "="

b = "black"
w = "white"

# Inverts the color. b -> w and w -> b
def invert(s):
    if s == b: return w
    if s == w: return b
    raise AssertionError("Not a valid color!")

## Python Utility Stuff
def Contains(array, element):
        filter = [n for n in array if n == element]
        return len(filter)

# Threshold is like a thickened boundary around the boundary of the unit circle
def isInUnitCircle(X, threshold = 0.05):
    # If the threshold is too close to 1 there might be infinite loop shenanigans
    assert threshold < 1 - EPSILON
    return X[0] ** 2 + X[1] ** 2 < (1 - threshold) ** 2

# Generate a random point in the unit circle
def randomPointInUnitCircle(threshold = 0.05, scale = 1):
    x, y = -1, -1
    while not isInUnitCircle((x, y), threshold):
        x = np.random.rand()
        y = np.random.rand()
    return np.array([x, y], dtype = np.float64) * scale

# L2 distance of two points
def L2Dist(v1, v2): 
    return ((v2[1] - v1[1]) ** 2 + (v2[0] - v1[0]) ** 2) ** 0.5

# Return vector from X to its "projection" on line segment X1 X2
def lineSegProjection(X1, X2, X):
    A = X[0] - X1[0];
    B = X[1] - X1[1];
    C = X2[0] - X1[0];
    D = X2[1] - X1[1];

    dot = A * C + B * D;
    len_sq = C * C + D * D;
    param = -1;
    if len_sq != 0: param = dot / len_sq;

    if param < 0: R = X1
    elif param > 1: R = X2
    else: R = X1 + param * (X2 - X1)
    return X - R

# Generate a given point on the circle with given radius and a parametrization 0 \le t \le 1
def Circle(radius, t):
    arr = np.array([radius * np.cos(2 * PI * plotOffsetAngle - 2 * PI * t), radius * np.sin(2 * PI * plotOffsetAngle - 2 * PI * t)], dtype = np.float64)
    if abs(np.sum(arr ** 2) - radius ** 2) > EPSILON:
        print("We have a problem: circle seems to be outputing wrong stuff: ", np.sum(arr ** 2), radius)
    return arr

# Check if line segment AB and CD intersect
def intersect(A, B, C, D):
    # Equation of line AB is A + t(B-A) and CD is C + t(D-C)
    BAx, BAy = B - A
    CDx, CDy = C - D
    CAx, CAy = C - A
    det = BAx * CDy - CDx * BAy
    if det == 0:
        return False
    t1 = (CAx * CDy - CDx * CAy) / det
    t2 = (BAx * CAy - CAx * BAy) / det
    print(t1, t2)
    if t1 > 1 or t2 > 1 or t1 < -EPSILON or t2 < -EPSILON:
        # In practice a point will not lie exactly on another line segment but just in case, we would want to untangle that as well
        return False
    return True

# Stretch a line segment from its midpoint
def stretch(A, B, stretch_factor = 0, constant_pad = 0.1):
    A = np.array(A, dtype = np.float64)
    B = np.array(B, dtype = np.float64)
    # Stretch a line by a given proportion
    v = B - A
    if stretch_factor == 0:
        # Calculate using the constant pad
        stretch_factor = max(0, 1 - (2 * constant_pad) / L2Dist(A, B))
    new_A = A + (0.5 - 0.5 * stretch_factor) * v
    new_B = A + (0.5 + 0.5 * stretch_factor) * v
    new_A = np.round(new_A, 5)
    new_B = np.round(new_B, 5)
    return tuple(new_A), tuple(new_B)

# Special invariant objects
# t: type (1: J^p_q, 2: J_pqr, 3: J^pqr, 4: J^pq_rs)
class J:
    def __init__(self, top = (), bottom = ()):
        self.top = tuple(sorted(top))
        self.bottom = tuple(sorted(bottom))
        if len(top) == 1 and len(bottom) == 1:
            self.t = 1
        elif len(top) == 0 and len(bottom) == 3:
            self.t = 2
        elif len(top) == 3 and len(bottom) == 0:
            self.t = 3
        elif len(top) == 2 and len(bottom) == 2:
            self.t = 4
        else:
            # Represents a zero tensor
            self.t = 0

    # Overload == operator for good measures, just in case we need to define other properties later
    def __eq__(self, J2):
        if type(J2) != type(self): return False
        return set(self.top) == set(J2.top) and set(self.bottom) == set(J2.bottom)

    def __lt__(self, J2):
        if self.t < J2.t: return True
        if tuple(sorted(self.top)) < tuple(sorted(J2.top)): return True
        if tuple(sorted(self.bottom)) < tuple(sorted(J2.bottom)): return True
        return False

    def __str__(self):
        if self.t == 0:
            return "0"
        if self.t == 1:
            return f"J^{{{self.top[0]}}}_{{{self.bottom[0]}}}"
        if self.t == 2:
            return f"J_{{{self.bottom[0]}, {self.bottom[1]}, {self.bottom[2]}}}"
        if self.t == 3:
            return f"J^{{{self.top[0]}, {self.top[1]}, {self.top[2]}}}"
        if self.t == 4:
            return f"J^{{{self.top[0]}, {self.top[1]}}}_{{{self.bottom[0]}, {self.bottom[1]}}}" 
        return "NOT AN INVARIANT"
    
    def Export(self):
        if self.t == 0:
            return "J()"
        if self.t == 1:
            return f"J(({self.top[0]},), ({self.bottom[0]},))"
        if self.t == 2:
            return f"J((), {self.bottom})"
        if self.t == 3:
            return f"J({self.top}, ())"
        if self.t == 4:
            return f"J({self.top}, {self.bottom})" 
        return "NOT AN INVARIANT"
    
    # I got messed up with my implementations lol
    def copy(self):
        return J(self.top, self.bottom)

    def Copy(self):
        return self.copy()

# Wapper for a product of J invariants
class InvariantTerm:
    def __init__(self, equation: list):
        self.equation = equation
        self.isZero = False
    
    #TODO add export to latex
    def __str__(self):
        st = [str(s) for s in self.equation]
        return "Equation " + '    '.join(st)

    # Replace elements in list
    def replace(self, find: J, replaceBy: list, verbose : bool = False, eqn_name : str = ""):
        lis = self.equation
        replaced = False
        # Rather unpythonic solution but 1. I dont care 2. it works
        # But you probably wont get the original list back :(
        # Also we want to remove terms one by one. Everything is implemented recursively anyway so it is slow but it is fine
        for i in range(len(lis)):
            if i >= len(lis): break
            if lis[i] == find:
                replaced = True
                if verbose:    print("Triggered replacement by " + eqn_name)
                if i + 1 == range(len(lis)): lis[-1:] = replaceBy
                else: lis[i:i+1] = replaceBy
                break
        self.equation = lis
        return replaced
    
    def __eq__(self, eqn):
        if type(self) != type(eqn): return False
        if len(self.equation) != len(eqn.equation): return False
        self_lis = []
        for j in self.equation: self_lis.append(str(j))
        eqn_list = []
        for j in eqn.equation: eqn_list.append(str(j))
        if sorted(self_lis) == sorted(eqn_list): return True
        return False

    # This makes stuff indexable
    def __iter__(self):
        self.__n = 0
        return self
    
    def __next__(self):
        a = self.__n
        if self.__n >= len(self.equation):
            raise StopIteration
        self.__n += 1
        return self.equation[a]
    
    def copy(self):
        return InvariantTerm([j.copy() for j in self.equation])
    
    def __getitem__(self, key):
        return self.equation[key]
    
    def __setitem__(self, key, val):
        assert type(val) == type(J())
        self.equation[key] = val

# A three InvariantTerm exchange relation toTerms means "arrow from x to _" and from terms means "arrow to x from _"
class SkeinRelation:
    def __init__(self, exchangeTerm: list, toTerms: list, fromTerms: list, signature: str = "", original: bool = True):
        # Exchange term is the xx' in a typical cluster exchange relation
        # toTerms is one that exchange term have arrow pointing towards
        # fromTerms is one that have arrow pointing to exchange term
        eq = ["", equal, "", plus, ""]
        eq[-1:] = fromTerms
        eq[2:3] = toTerms
        eq[0:1] = exchangeTerm
        self.equation = eq
        self.ex = InvariantTerm(exchangeTerm)
        self.to = InvariantTerm(toTerms)
        self.fr = InvariantTerm(fromTerms)
        self.isZero = exchangeTerm == [] and toTerms == [] and fromTerms == []
        self.signature = signature
        # Signature is a string that gets printed along the relation for debug purposes
        # Original denotes if the relation is from the original triangulation
        self.original = original

        if not self.isZero and len(exchangeTerm) != 2:
            print(*exchangeTerm)
            raise AssertionError("Length must be 2")

    def __str__(self):
        st = [s if (s == plus or s == equal) else str(s) for s in self.equation]
        return "Exchange Relation " + ' '.join(st) + " " + self.signature

    def __eq__(self, rel):
        if type(rel) != type(self): return False
        # The exchange mutation term matters, so the ordering of this part must be correct
        if not self.ex[0] == rel.ex[0] or not self.ex[1] == rel.ex[1]: return False
        return self.to == rel.to and self.fr == rel.fr

    def Export(self, mode = "math"):
        if mode == "math":
            st = [s if (s == plus or s == equal) else str(s) for s in self.equation]
            return "\\[" + ' '.join(st) + " \\hspace{8pt} \\text{" + self.signature + "} \\]"
        elif mode == 'python':
            ex, to, fr = [j.Export() for j in self.ex], [j.Export() for j in self.fr], [j.Export() for j in self.fr]
            exst, frst, tost = ", ".join(ex), ", ".join(fr), ", ".join(to)
            st = f"SkeinRelation([{exst}], [{tost}], [{frst}], signature = \"{self.signature})\", original = {self.original})"
            return st

    def swap(self):
        self.to, self.fr = self.fr, self.to
        eq = ["", equal, "", plus, ""]
        eq[-1:] = self.fr.equation
        eq[2:3] = self.to.equation
        eq[0:1] = self.ex.equation
        # self.signature += " \t\t swapped"
        self.equation = eq
    
    def copy(self):
        exc = self.ex.copy()
        fro = self.fr.copy()
        too = self.to.copy()
        return SkeinRelation(exc.equation, too.equation, fro.equation, self.signature, self.original)

# A node on a tensor diagram
class TensorNode:
    def __init__(self, id = 0, isBoundary = False, color = b, connected = [-1, -1, -1], coordinates = np.array([0,0], dtype = np.float64)):
        assert color == b or color == w

        # Use two different instantiations for boundary nodes and internal nodes
        if connected == [-1, -1, -1]:
            if isBoundary: self.connected = []
            else: self.connected = [-1, -1, -1]
        else: self.connected = connected
        
        self.isBoundary = isBoundary
        self.color = color
        self.coordinates = coordinates
        self.id = id
        self.force = np.array([0,0], dtype = np.float64)
        self.boundaryDistance = 0
        self.explored = False
        self.parent = None

    def __str__(self):
        st = "Boundary " if self.isBoundary else "Internal "
        return st + "node id: " + str(self.id) + ", Color: " + str(self.color) + ", Connected to " + str(self.connected)

    @staticmethod
    def __append__(node1, node2, ifc: bool):
        if node1.isBoundary or ifc: 
            node1.connected.append(node2.id)
        else:
            for i in range(4):
                # i == 3 means it has already overflowed
                if i == 3:
                    print(node1)
                    raise AssertionError("Node connections already full!")
                
                # if the spot is -1 then we get something nice yay
                if node1.connected[i] < 0:
                    node1.connected[i] = node2.id
                    break

    def connect(self, node, ignore_full_connection = False):
        # print(f"Connecting {self} and {node}", end = "\n")
        assert type(self) == type(node)
        TensorNode.__append__(self, node, ignore_full_connection)
        TensorNode.__append__(node, self, ignore_full_connection)
    
    def Reset(self):
        self.force = np.array([0,0], dtype = np.float64)
        self.explored = False
        self.parent = None
        self.boundaryDistance = 0

    def coordinate(self, precision = 5):
        return (round(self.coordinates[0], precision) , round(self.coordinates[1], precision))

# A type a, b tensor
class TensorDiagram:
    def __init__(self, signature):
        self.signature = signature
        self.a = signature.count(b)
        self.b = signature.count(w)
        self.l = len(signature)
        
        # Generate the sign for latex
        st = "["
        lis = ["\\bullet " if t == b else "\\circ " for t in signature]
        st += ", ".join(lis)
        st += "]"
        self.sign = st


        # Generate boundary nodes
        self.nodes = []
        for i in range(self.l):
            # We want to construct the nodes clockwise from the top
            self.AddNode(signature[i], coordinates = Circle(1, 2 * plotOffsetAngle + i / self.l), isBoundary = True)
        
        # Used for printing
        self.scale = 3
        self.shift = (0,0)
        self.dotsize = 0.8


    # Get the color of the node at pos
    def color(self, pos: int, boundary = True):
        node = self.GetNodeFromID(pos, boundary)
        return node.color



    # Get the next vacant id for nodes
    def GetNextNodeID(self):
        i = 0
        while i < len(self.nodes) + 1:
            # Loop through nodes to find the smallest i value that hasnt been used yet
            filter = [n for n in self.nodes if n.id == i]
            if len(filter) == 0:
                return i
            i += 1
        raise RuntimeError("Somehow the GetNextNodeID could not get the next node id. The loop overflowed")
    

    # Given ID give us back the node. Boundary: only get boundary nodes
    def GetNodeFromID(self, id: int, boundary: bool):
        # Sanity Check: if id is -1 then return nothing
        if id < 0: raise AssertionError("ID must be > 0")

        # If you intend on getting only the boundary vertices then I will take mod for you
        if boundary: id %= self.l
        
        filter = [n for n in self.nodes if n.id == id]
        if len(filter) == 0:
            print(id)
            print(*self.nodes, sep = "\n")
            raise AssertionError("No node with such id exist!")
        if len(filter) > 1:
            print(*self.nodes, sep = "\n")
            raise AssertionError("Something went wrong! There are multiple nodes with given id")
        return filter[0]


    # Add a new node with a given color and connected nodes
    # Returns the new node
    def AddNode(self, color: str, isBoundary = False, connected = [-1, -1, -1], coordinates = (0,0)):
        id = self.GetNextNodeID()
        node = TensorNode(id, isBoundary, color, connected, coordinates)
        self.nodes.append(node)
        return id, node


    # Adds a catepillar invariant which are building blocks of the J invariants. Returns the node right above the starting boundary node
    # If color = b then it is a black tree aka \lambda^j, otherwise it is a white tree aka \lambda_k
    def AddSpecialInvariant(self, startNode, color: str):
        # Sanity check
        if color != b and color != w: raise AssertionError("Color must be black or white!")

        # Get Boundary node first
        boundaryNode = self.GetNodeFromID(startNode, True)

        # If the color of the tree and the color of the boundary node is opposite color then return the starting node
        # Since there is nothing to construct
        # Otherwise construct the tree node
        if color != boundaryNode.color: return boundaryNode

        id, treeNode = self.AddNode(invert(boundaryNode.color), coordinates = randomPointInUnitCircle(scale = 1e-2))
        # print("New node id:" + str(id))
        # print("Tree node connections:" + str(treeNode.connected))
        # print("Boundary node connections:" + str(boundaryNode.connected))
        treeNode.connect(boundaryNode)
        
        # if the next color is same as this color, then connect to both nodes and bounce, otherwise do recursion magic
        nextNode = self.GetNodeFromID(startNode + 1, True)
        if nextNode.color == boundaryNode.color: 
            treeNode.connect(nextNode)
        else: 
            treeNode.connect(self.AddSpecialInvariant(startNode + 1, invert(color)))
        return treeNode

    # Takes mod on everything
    # Alternative constructors for J
    def J1(self, p, q):
        return J((p % self.l,), (q % self.l,))

    def J2(self, p, q, r):
        return J((), (p % self.l, q % self.l, r % self.l))


    def J3(self, p, q, r):
        return J((p % self.l, q % self.l, r % self.l), ())

    def J4(self, p, q, r, s):
        return J((p % self.l, q % self.l), (r % self.l, s % self.l))

    # If it is one term, then the method removes everything upon seeing a zero, otherwise it sweeps all the zeros under the rug and pretend nothing happened
    def __factorRecursive__(self, eqn: InvariantTerm, isOneTerm, verbose):
        for p in range(self.l):
            # Trivial cases:
            if eqn.replace(self.J1(p, p), [J()], verbose, "6.4 trivial"): break
            # if self.signature == [w, b, w, b, w]: eqn.replace(self.J1(1, 0), [self.J1(0,3), self.J1(2,4)], verbose, "Special case wbwbw")
            # if self.signature == [b, w, b, w, b]: eqn.replace(self.J1(0, 1), [self.J1(3, 0), self.J1(4,2)], verbose, "Special case bwbwb")

            # p is white cases
            if self.color(p) == w:
                if eqn.replace(self.J1(p+1, p), [J()], verbose, "6.4.0w"): break
                if self.color(p+1) == b:
                    eqn.replace(self.J1(p+2, p), [self.J1(p, p+2), self.J1(p+2, p+1)], verbose, "6.4.1w" + ", p = " + str(p))
                    if self.color(p+2) == w:
                        eqn.replace(self.J1(p+3, p), [self.J1(p+3, p+1), self.J1(p, p+2)], verbose, "6.4.2w" + ", p = " + str(p))
                else: # self.color(p + 1) == w
                    eqn.replace(self.J1(p+2, p), [self.J1(p, p+1)], verbose, "EQ1w" + ", p = " + str(p))
                for q in range(self.l):
                    if q == p: continue
                    eqn.replace(self.J2(p, p+1, q), [self.J1(p, p+1), self.J1(p+1, q)], verbose, "6.4.3w" + ", p = " + str(p) + ", q = " + str(q))
                    eqn.replace(self.J3(p, p+1, q), [self.J1(q, p)], verbose, "EQ3w" + ", p = " + str(p) + ", q = " + str(q))
                    if self.color(p + 1) == b:
                        eqn.replace(self.J2(p, p+2, q), [self.J1(p, p+2), self.J1(p+1, q)], verbose, "6.4.4w" + ", p = " + str(p) + ", q = " + str(q))
                        eqn.replace(self.J4(p, q, p+1, p+2), [self.J1(q, p)], verbose, "EQ4w" + ", p = " + str(p) + ", q = " + str(q))
                    for r in range(self.l):
                        if r == p or r == q: continue
                        eqn.replace(self.J4(q, r, p, p+1), [self.J1(p, p+1), self.J3(p+1, q, r)], verbose, "6.4.5w" + ", p = " + str(p) + ", q = " + str(q)    + ", r = " + str(r))
                        eqn.replace(self.J4(p + 1, q, r, p), [self.J1(p + 1, r), self.J1(q, p)], verbose, "6.4.6w" + ", p = " + str(p) + ", q = " + str(q)    + ", r = " + str(r))
                        eqn.replace(self.J4(p, p+1, q, r), [self.J2(p, q, r)], verbose, "EQ2w" + ", p = " + str(p) + ", q = " + str(q)    + ", r = " + str(r))
                        if self.color(p+1) == b:
                            eqn.replace(self.J4(q, r, p, p+2), [self.J1(p, p+2), self.J3(p+1, q, r)], verbose, "6.4.7w" + ", p = " + str(p) + ", q = " + str(q)    + ", r = " + str(r))

            # p is black cases
            elif self.color(p) == b:
                if eqn.replace(self.J1(p, p+1), [J()], verbose, "6.4.0b" + ", p = " + str(p)): break
                if self.color(p+1) == w:
                    eqn.replace(self.J1(p, p+2), [self.J1(p+2, p), self.J1(p+1, p+2)], verbose, "6.4.1b" + ", p = " + str(p))
                    if self.color(p+2) == b:
                        eqn.replace(self.J1(p, p+3), [self.J1(p+1, p+3), self.J1(p+2, p)], verbose, "6.4.2b" + ", p = " + str(p))
                else:    # self.color(p + 1) == b
                    eqn.replace(self.J1(p, p+2), [self.J1(p+1, p)], verbose, "EQ1b" + ", p = " + str(p))
                for q in range(self.l):
                    if q == p: continue
                    eqn.replace(self.J3(p, p+1, q), [self.J1(p+1, p), self.J1(q, p+1)], verbose, "6.4.3b" + ", p = " + str(p) + ", q = " + str(q))
                    eqn.replace(self.J2(p, p+1, q), [self.J1(p, q)], verbose, "EQ3b" + ", p = " + str(p) + ", q = " + str(q))
                    if self.color(p + 1) == w:
                        eqn.replace(self.J3(p, p+2, q), [self.J1(p+2, p), self.J1(q, p+1)], verbose, "6.4.4b" + ", p = " + str(p) + ", q = " + str(q))
                        eqn.replace(self.J4(p+1, p+2, p, q), [self.J1(p, q)], verbose, "EQ4b" + ", p = " + str(p) + ", q = " + str(q))
                    for r in range(self.l):
                        if r == p or r == q: continue
                        eqn.replace(self.J4(p, p + 1, q, r), [self.J1(p+1, p), self.J2(p+1, q, r)], verbose, "6.4.5b" + ", p = " + str(p) + ", q = " + str(q)    + ", r = " + str(r))
                        eqn.replace(self.J4(r, p, p + 1, q), [self.J1(r, p + 1), self.J1(p, q)], verbose, "6.4.6b" + ", p = " + str(p) + ", q = " + str(q)    + ", r = " + str(r))
                        eqn.replace(self.J4(q, r, p, p+1), [self.J3(p, q, r)], verbose, "EQ2b" + ", p = " + str(p) + ", q = " + str(q)    + ", r = " + str(r))
                        if self.color(p+1) == w: 
                            eqn.replace(self.J4(p, p+2, q, r), [self.J1(p+2, p), self.J2(p+1, q, r)], verbose, "6.4.7b" + ", p = " + str(p) + ", q = " + str(q)    + ", r = " + str(r))

        # Remove all zeros from the list of terms
        if isOneTerm:
            for j in eqn.equation:
                if j == J():
                        eqn.isZero = True
                        eqn.equation = []
                        return eqn
        else:
            emptyj = J()
            eqn.equation = [j for j in eqn.equation if not j == emptyj]
        return eqn

    def Factor(self, equation: InvariantTerm, isOneTerm, maxIter = 15, verbose = False):
        i = 1
        if verbose: 
            print("Factoring: " + str(equation))
            print("Iteration: {}".format(i))
        f = self.__factorRecursive__(equation, isOneTerm, verbose)
        last = str(f)

        if verbose: 
            print(last)
        while i < maxIter:
            # factor
            f = self.__factorRecursive__(equation, isOneTerm, verbose)
            current = str(f)
            if current == last: 
                break

            # Update
            last, i = current, i + 1

            if verbose: 
                print("Iteration: {}".format(i))
                print(current)
        
        # Print final result
        if verbose: 
            print("Result: " + str(equation) + "\n")
        return equation

    # Add a J invariant to the cluster
    def AddJInvariant(self, inv: J):
        # J_p^q invariant
        if inv.t == 1: 
            self.AddSpecialInvariant(inv.bottom[0], w).connect(self.AddSpecialInvariant(inv.top[0], b))
        # J_pqr invariant
        if inv.t == 2:
            t, node = self.AddNode(w)
            node.connect(self.AddSpecialInvariant(inv.bottom[0], w))
            node.connect(self.AddSpecialInvariant(inv.bottom[1], w))
            node.connect(self.AddSpecialInvariant(inv.bottom[2], w))
        # J^pqr invariant
        if inv.t == 3:
            t, node = self.AddNode(b)
            node.connect(self.AddSpecialInvariant(inv.top[0], b))
            node.connect(self.AddSpecialInvariant(inv.top[1], b))
            node.connect(self.AddSpecialInvariant(inv.top[2], b))
        # J_pq^rs invariant
        if inv.t == 4:
            t, nodeb = self.AddNode(b)
            t, nodew = self.AddNode(w)
            nodeb.connect(self.AddSpecialInvariant(inv.top[0], b))
            nodeb.connect(self.AddSpecialInvariant(inv.top[1], b))
            nodew.connect(self.AddSpecialInvariant(inv.bottom[0], w))
            nodew.connect(self.AddSpecialInvariant(inv.bottom[1], w))
            nodeb.connect(nodew)

    # Process Skein equation
    def Skein(self, skein: SkeinRelation, verbose = False):
        if verbose: print("\n=================================================================================\nUnprocessed equation: " + skein.Export(mode = "python"))
        fr = self.Factor(skein.fr, True, verbose = verbose)
        to = self.Factor(skein.to, True, verbose = verbose)
        ex = self.Factor(skein.ex, True, verbose = verbose)

        # That means all 3 terms might have some common invariant that we need to factor out and remove
        extra_ex = []
        for j in ex:
            extra_ex.append(J(j.top, j.bottom))
        
        for j in extra_ex:
            if (Contains(fr, j) or fr.isZero) and (Contains(fr, j) or to.isZero):
                if verbose:
                    print("Removing term " + str(j) + " for: \t", end = "")
                    print(ex, fr, to)
                ex.replace(j, [], verbose, "factor common terms")
                fr.replace(j, [])
                to.replace(j, [])

        try:
            if fr.isZero:
                assert to == ex
                return SkeinRelation([], [], [])
        
            if to.isZero:
                assert fr == ex
                return SkeinRelation([], [], [])

            if ex.isZero:
                assert fr.isZero and to.isZero
                return SkeinRelation([], [], [])

            assert len(ex.equation) == 2
        except:
            print("\n\n Error!")
            print(skein)
            print()
            print(ex, to, fr)
            raise AssertionError("Terms did not factor nicely please check the code")
        
        eq = SkeinRelation(ex.equation, to.equation, fr.equation, skein.signature)

        if verbose: 
            print("Processed equation: ", end = "")
            print(eq.Export(mode = "python"))
            print("\n\n")
        return eq

    #################################### Helper functions for printing the diagram ####################################

    #Reset All nodes
    def ResetAll(self):
        for n in self.nodes:
            n.explored = False
            n.parent = None

    # Run a BFS algorithm to get the distance to boundary for each node
    def __GetClosestBoundary__(self, root: TensorNode):
            q = [root]
            root.explored = True
            while len(q) > 0:
                v = q.pop(0)
                if v.isBoundary:
                    return v
                for neighbour in v.connected:
                    if neighbour < 0: continue
                    neighbourNode = self.GetNodeFromID(neighbour, boundary = False)
                    if not neighbourNode.explored:
                        neighbourNode.parent = v
                        neighbourNode.explored = True
                        q.append(neighbourNode)

    def __SumDist__(self, node, secondpass = False):
        sum = 0
        # print("Current Node: " + str(node) + str(node.boundaryDistance))
        # Add edge optimization
        for no in node.connected:
            # if id is -1 that means it is an empty node
            if no < 0: continue

            n = self.GetNodeFromID(no, boundary = False)

            if n.boundaryDistance >= 0 and n.boundaryDistance < node.boundaryDistance:
                sum += L2Dist(node.coordinates, n.coordinates)
        
        # Add vertex optimization - if too close to another vertex then gg
        if secondpass:
            for vertex in self.nodes:
                if vertex == node:
                    continue
                if L2Dist(node.coordinates, vertex.coordinates) < vertexThreshold: sum += vertexRepellance * len(self.nodes)

                # Add edge optimization - if too close to another edge then gg
                for n in vertex.connected:
                    if n < 0 or n == node.id: continue
                    node2 = self.GetNodeFromID(n, False)

                    # Compare and see if the edge weight is already added                
                    # If it is not an edge that connectes to itself or empty, then add the edge weight
                    direction = lineSegProjection(vertex.coordinates, node2.coordinates, node.coordinates)
                    if np.linalg.norm(direction) < edgeThreshold: sum += edgeRepellance * len(self.nodes)
        return sum

    def OptimizeNode(self, maxDist, divs, secondpass : bool = False):
        for i in range(maxDist + 1):
            radius = 1 / (i + 1) ** 0.5
            filter = [node for node in self.nodes if node.boundaryDistance == i]
            # print("Distance: " + str(i))
            # print(*filter, sep = "\n")

            # Optimize each node cuz screw this
            for node in filter:
                if node.isBoundary: 
                    continue
                t = np.arange( 0, divs )
                s = np.zeros((divs, 2))
                s[:, 0] = radius * np.cos(t/divs * 2 * PI)
                s[:, 1] = radius * np.sin(t/divs * 2 * PI)
                result = np.ones((divs,))
                for i in range(divs):
                    node.coordinates = s[i]
                    result[i] = self.__SumDist__(node, secondpass)
                node.coordinates = s[np.argmin(result)]

    # Puts the nodes on a fixed radius and optimize the positions
    def OptimizeDiagram(self, divs = 500):
        maxDist = 0
        # Run a BFS algorithm to get the distance to boundary for each node
        for root in self.nodes:
            if root.isBoundary:
                root.boundaryDistance = 0 
                continue

            closest = self.__GetClosestBoundary__(root)

            # Get the closest path
            path = []
            while closest != None:
                path.append(closest)
                closest = closest.parent
            # print("Current Node: " + str(root))
            # print("Path = ", end = "")
            # print(*path, sep = " \n")
            dist = len(path)
            if dist > maxDist: maxDist = dist
            root.boundaryDistance = dist

            self.ResetAll()
        
        self.OptimizeNode(maxDist, divs, False)
        self.OptimizeNode(maxDist, divs, True)

        # Reset the nodes for future optimization
        for n in self.nodes:
            n.Reset()
            
    def __generateStrContents__(self):
        st = ""
        # This portion draw the lines
        drawnLines = []
        for n in self.nodes:
            for id in n.connected:
                # if id is -1 that means it is an empty node
                if id < 0: continue
                toNode = self.GetNodeFromID(id, False)

                # Check and see if node is already drawn - check the drawn Lines list and see if the set {n.id, to.id} is already registered
                if len([d for d in drawnLines if d == set([n.id, id])]) > 0: continue

                # Draw the line - append tikz text
                fromCoord = (round(n.coordinate()[0] + self.shift[0], 5), round(n.coordinate()[1] + self.shift[1], 5))
                toCoord = (round(toNode.coordinate()[0] + self.shift[0], 5), round(toNode.coordinate()[1] + self.shift[1], 5))
                st += f"\n\t\\draw[black, thick] {fromCoord} -- {toCoord};"
                drawnLines.append(set([n.id, id]))
        
        # This portion draws the nodes
        for n in self.nodes:
            fromCoord = (round(n.coordinate()[0] + self.shift[0], 5), round(n.coordinate()[1] + self.shift[1], 5))
            if n.color == w:
                st += f"\n\t\\filldraw[color=black, fill=white] {fromCoord} circle ({self.dotsize}pt);"
            elif n.color == b:
                st += f"\n\t\\filldraw[color=black] {fromCoord} circle ({self.dotsize}pt);"
        
        # Adds the numbering labels
        for i in range(self.l):
            c = Circle(1.2, 2 * plotOffsetAngle + i / self.l)
            coord = (round(c[0] + self.shift[0], 5), round(c[1] + self.shift[1], 5))
            st += f"\n\t\\node[] at {coord} {{{i}}};"
        return st

    def __str__(self):
        self.OptimizeDiagram()
        st = "\\centering\n\\begin{{tikzpicture}}[scale = {}]\n\t\\filldraw[color=black, fill=white!0, thin]{} circle (1);".format(self.scale, self.shift)
        st += self.__generateStrContents__()
        st += "\n\\end{tikzpicture}\n\\par\n\\raggedright"
        return st

# A triangulation is a list of tuples containing the diagonals. We do not have a way to verify if your triangulation is a bona fide triangulation.
# Don't cheat :)
# Also we use zero count here, because computer uses zero count.
class Triangulation:
    def __init__(self, num_sides: int, edges: list):
        self.l = num_sides
        if num_sides == 0 and edges == []:
            self.trivial = True
            return

        addedEdge = []
        for edge in edges:
            if (edge[1] % num_sides == (edge[0] + 1) % num_sides or edge[1] % num_sides == (edge[0] - 1) % num_sides ) or Contains(addedEdge, set(edge)):
                edges.remove(edge)
                continue
            addedEdge.append(set((edge[0] % self.l, edge[1] % self.l)))

        assert len(edges) == num_sides - 3

        diagonals = []
        for edge in edges:
            diagonals.append(edge)
        
        # We also add in the edges
        for i in range(num_sides - 1):
            edges.append((i, i + 1))
        edges.append((num_sides - 1, 0))

        # The extended version contains the reverse edges also so each edge is basically added twice
        extendedEdges = []
        for edge in edges:
            extendedEdges.append((edge[1], edge[0]))
            extendedEdges.append(edge)

        # Now generate all clockwise 3 cycles using dft algorithm
        # The normal version just contains whatever
        # The extended version may contain different encodings of the same triangle (p, q, r) but p, r is always a diagonal
        trigs_set = []
        trigs = []
        extendedTrigs = []
        # Since we only need 3 cycles we can skip the recursion and just use 3 layers of for loop
        # We can slightly optimize the diagram by not checking the last 2
        for i in range(num_sides - 2):
            i_connections = [v for v in extendedEdges if v[0] == i]
            for i_con in i_connections:
                j_connections = [v for v in extendedEdges if v[0] == i_con[1]]
                for j_con in j_connections:
                    k_connections = [v for v in extendedEdges if v[0] == j_con[1]]
                    for k_con in k_connections:
                        if (k_con[0], i) in extendedEdges:

                            # If this is already added then the extended version is already taken care of
                            if Contains(trigs_set, set((i, i_con[1], j_con[1]))): continue
                            trigs_set.append(set((i, i_con[1], j_con[1])))

                            # Check all 3 cases to deal with the oriented extendedTrig
                            candidate = tuple(sorted((i, i_con[1], j_con[1])))
                            trigs.append(candidate)
                            
                            if not abs(candidate[1] - candidate[0]) == 1:
                                extendedTrigs.append((candidate[1], candidate[2], candidate[0]))
                            if not abs(candidate[2] - candidate[1]) == 1:
                                extendedTrigs.append((candidate[2], candidate[0], candidate[1]))
                            if not(abs(candidate[0] - candidate[2]) == 1 or abs(candidate[0] - candidate[2]) == num_sides - 1):
                                extendedTrigs.append(candidate)

        four_cycles = []
        ext_4 = []
        # Now also generate all the 4 cycles
        for t in extendedTrigs:
            for s in extendedTrigs:
                if t == s or set([t[0], t[2]]) != set([s[0], s[2]]):
                    continue
                ext_4.append((t[0], t[1], s[0], s[1]))
                # Check if we already logged the
                filter = [c for c in four_cycles if set(c) == set([t[0], t[1], s[0], s[1]])]
                if len(filter) > 0: continue
                # Either case this is correctly oriented
                four_cycles.append((t[0], t[1], s[0], s[1]))
        
        assert len(edges) == num_sides + len(diagonals)
        assert len(trigs) == num_sides - 2
        assert len(four_cycles) == num_sides - 3
        assert len(ext_4) == 2 * (num_sides - 3)
        assert len(diagonals) == num_sides - 3

        self.edges = edges                                    # undirected edges of the triangulation: tuple
        self.extendedEdges = extendedEdges    # directed edges of the triangulation with both direction : tuple
        self.extendedTrigs = extendedTrigs    # All triangle pqr where pr is a diagonal: tuple
        self.trigs = trigs                                    # Some clockwise oriented triangle: tuple
        self.four_cycles = four_cycles            # All 4 cycles of some form: tuple
        self.extended_four_cycles = ext_4     # list of unfiltered 4 cycles: tuple
        self.diagonals = diagonals                    # the list of diagonals: tuple
        self.trivial = False

    # Generates a "triangulation" thats a list of diagonals with the given list
    @staticmethod
    def GenerateTriangulation(lis: list):
        n = len(lis)
        tri = [(lis[0], lis[2]), (lis[2], lis[-1])]
        for i in range(3, int(np.floor(n / 2)) + 1):
            tri.append((lis[i], lis[2 - i]))
            tri.append((lis[i], lis[1 - i]))
        return tri

    def __str__(self):
        return str(self.diagonals)

# A J-invariant which also holds information about the cluster, in particular who is pointing to it and away from it in its cluster
class ClusterVariable(J):
    def __init__(self, copy: J, cluster: bool = False):
        super().__init__(copy.top, copy.bottom)
        self.j = copy
        # List of J* to check against
        self.fromTerms = []
        self.toTerms = []
        self.isFrozen = not cluster

    def reset(self):
        self.fromTerms = []
        self.toTerms = []
    
    def __eq__(self, J2):
        return super().__eq__(J2)

# Generate the QMU file from exchange matrix
def GetQMU(B_matrix : np.ndarray, fileName, save_file = True):
    assert B_matrix.shape[0] == B_matrix.shape[1]
    num_vertices = B_matrix.shape[0]
    st = "//Number of points\n"
    st += str(num_vertices)
    st += "\n//Vertex radius\n9\n//Labels shown\n1\n//Matrix\n"
    st += str(num_vertices) + " " + str(B_matrix.shape[1])
    st += "\n"
    for i in range(num_vertices):
        for j in range(num_vertices):
            st += str(B_matrix[i, j])
            st += " "
        st += "\n"
    st += "//Points\n"
    for i in range(num_vertices):
        coord = Circle(420, i/num_vertices)
        st += "9 {} {} \n".format(round(coord[0], 1), round(coord[1], 1))
    st += "//Cluster is null\n//Historycounter\n-1\n//History"
    if not save_file: return st

    fn = '{}.qmu'.format(fileName)
    with open(fn, 'w') as f:
        f.write(st)
    return st

# Cluster algebra class
class ClusterAlgebra:
    # Constructor is called by giving the initial seed
    def __init__(self, cluster_variables: list, frozen_variables: list, exchange_relations: list):
        # Rank of cluster algebra
        self.rank = len(cluster_variables)
        m = self.rank + len(frozen_variables)

        # A list of generators for cluster, frozen and exchange relations
        self.clusterVariables = []

        # Enumerate objects have (index, object)
        for t in enumerate(cluster_variables):
            self.clusterVariables.append(t)
        
        self.frozenVariables = []
        for t in zip(range(self.rank, m), frozen_variables):
            self.frozenVariables.append(t)
        
        self.exchangeRelations = exchange_relations

        # Generate the (extended) exchange matrix
        B_matrix = np.zeros((m, m), dtype = int)
        for j, cv in self.clusterVariables + self.frozenVariables:
            # Loop through every row now :)
            for i, other in self.clusterVariables + self.frozenVariables:
                if cv.j == other.j: continue
                num_arrows_from_cv, num_arrows_to_cv = 0, 0
                # Loop through all the from terms in cluster variable and add the entry
                # cv, other are Generators but stuff in fromTerms are not
                for variable in cv.fromTerms:
                    if variable == other.j: num_arrows_from_cv += 1
                for variable in cv.toTerms:
                    if variable == other.j: num_arrows_to_cv += 1                
                B_matrix[i, j] = num_arrows_from_cv - num_arrows_to_cv
        
        self.full_exchange_matrix = B_matrix
        self.extended_exchange_matrix = B_matrix[:, :self.rank]
        self.exchange_matrix = B_matrix[: self.rank, :self.rank]

# Define mutations. We cheat by defining different behaviour for this function depending if we feed it a list or an integer
def Mutate(matrix, k):
    if type(k) == type([1, 2, 3]):
        for elem in k:
            matrix = Mutate(matrix, elem)
        return matrix
    elif type(k) == type(0):
        # Since we need to sort of update the whole matrix all at once we construct an identical one to keep track of stuff
        # return by copy
        B = np.array(matrix, dtype = int)
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                # Equation 3.10 in Professor Ip's lecture notes
                B[i, j] = -matrix[i, j] if (k == i or k == j) else matrix[i, j] + 0.5 * (np.abs(matrix[i, k]) * matrix[k , j] + matrix[i, k] * np.abs(matrix[k, j]))
        return B
    else:
        raise TypeError("k must be an integer or a list!")

# A tensor diagram where you can feed it a triangulation and it will generate a cluster for you
class TriangulationDiagram(TensorDiagram):
    def __init__(self, signature):
        super().__init__(signature)
        tri = Triangulation.GenerateTriangulation(list(range(self.l)))
        self.triangulation = Triangulation(self.l, tri)
        self.generated = False

    def Triangulate(self, triangulation: Triangulation):
        assert self.l == triangulation.l
        self.triangulation = triangulation

    def PlotTriangulation(self):
        newDiagram = TriangulationDiagram(self.signature)
        newDiagram.Triangulate(self.triangulation)
        for edge in newDiagram.triangulation.edges:
            node1 = newDiagram.GetNodeFromID(edge[0], True)
            node2 = newDiagram.GetNodeFromID(edge[1], True)
            node1.connect(node2)
        print(newDiagram)

    # Write down all cluster variables
    def GenerateCluster(self, verbose = False, veryVerbose = False):
        if veryVerbose: verbose = True

        # Frozen Variables
        frozen = []
        for i in range(self.l):
            if self.color(i) == b:
                frozen.append(self.J1(i + 1, i))
            if self.color(i) == w:
                frozen.append(self.J1(i, i + 1))

        # Cluster Variables
        cv0, relations = [], []
        for edge in self.triangulation.extendedEdges:
            cv0.append(self.J1(*edge))

        for t in self.triangulation.trigs:
            cv0.append(self.J2(*t))

        # Helper function to factor all terms in a cluster variable
        def FactorAllTerms(cv):
            cv2 = []
            while len(cv) > 0:
                j = cv.pop(0)
                fac = self.Factor(InvariantTerm([j]), True)
                for term in fac.equation:
                    cv2.append(term)
                # Remove duplicates
                res = []
                [res.append(x) for x in cv2 if x not in res]
            return cv2

        # Now factor everything 
        cv = FactorAllTerms(cv0)
        i = 0
        while cv != cv0:
            cv0 = [j.Copy() for j in cv]
            cv = FactorAllTerms(cv)
            i += 1
        
        # Remove duplicates
        clustervars = []
        [clustervars.append(x) for x in cv if (x not in clustervars) and (x not in frozen)]

        # Reality check: Theorem 7.1 asserts there is exactly 2n-8 cluster variables
        assert len(clustervars) == 2 * (self.a + self.b) - 8
        clusterVariables = [ClusterVariable(x, cluster = True) for x in clustervars]
        frozenVariables = [ClusterVariable(x, cluster = False) for x in frozen]

        # Print information so far
        if verbose:
            print("Cluster Variables = ", end = " ")
            print(*clustervars)
            print("Frozen Variables = ", end = " ")
            print(*frozen)

        # Consider exchange relations. Check if diagonal ab is actually an edge
        def isEdge(a, b):
            return a == (b + 1) % self.l or b == (a + 1) % self.l
        
        # Add exchange relations to the big bag
        def AddRelation(relations: list, tri: Triangulation, original):
            # Handles equation 6.1
            for t in tri.trigs:
                p, q, r = t
                if isEdge(p, q) or isEdge(q, r): continue
                # Skein relation 6.1
                rel61 = self.Skein(SkeinRelation([self.J2(*t), self.J3(*t)], [self.J1(p, r), self.J1(r, q), self.J1(q, p)] ,[self.J1(p, q), self.J1(q, r), self.J1(r, p)], signature = "from 6.1 with t = " + str(t), original = original), verbose = veryVerbose)
                if not rel61.isZero: relations.append(rel61)
            
            for c in tri.extended_four_cycles:
                p, q, r, s = c
                # Skein relation 6.2 - 6.5
                rel62 = self.Skein(SkeinRelation([self.J1(r, p), self.J2(q, r, s)], [self.J1(r, q), self.J2(p, r, s)], [self.J1(r, s), self.J2(q, r, p)], signature = "from 6.2" + " with p, q, r, s = " + str((p, q, r, s)), original = original), verbose = veryVerbose)
                if not rel62.isZero: relations.append(rel62)
                
                # Check for one exposed edge. pr must be a diagonal so no need to check that
                if isEdge(p, q) or isEdge(q, r):
                    rel64 = self.Skein(SkeinRelation([self.J1(r, p), self.J4(p, q, r, s)], [self.J1(q, p), self.J1(p, r), self.J1(r, s)], [self.J2(p, r, s), self.J3(q, r, p)], signature = "from 6.4" + " with p, q, r, s = " + str((p, q, r, s)), original = original), verbose = veryVerbose)
                    rel65 = self.Skein(SkeinRelation([self.J1(p, r), self.J4(q, r, s, p)], [self.J1(q, r), self.J1(p, s), self.J1(r, p)], [self.J2(p, r, s), self.J3(q, r, p)], signature = "from 6.5" + " with p, q, r, s = " + str((p, q, r, s)), original = original), verbose = veryVerbose)
                    if not rel64.isZero: relations.append(rel64)
                    if not rel65.isZero: relations.append(rel65)

                # Check for two exposed edge
                if isEdge(p, q) and isEdge(q, r):
                    rel63 = SkeinRelation([], [], [])
                    if self.color(p, True) == w and self.color(p + 1, True) == b:
                        rel63 = self.Skein(SkeinRelation([self.J1(p, p+2), self.J1(s, p+1)], [self.J1(p, p+1), self.J1(s, p+2)], [self.J1(s, p)], signature = "from 6.3w" + " with p, q, r, s = " + str((p, q, r, s)), original = original), verbose = veryVerbose)
                    if self.color(p, True) == b and self.color(p + 1, True) == w:
                        rel63 = self.Skein(SkeinRelation([self.J1(p + 2, p), self.J1(p+1, s)], [self.J1(p+1, p), self.J1(p+2, s)], [self.J1(p, s)], signature = "from 6.3b" + " with p, q, r, s = " + str((p, q, r, s)), original = original), verbose = veryVerbose)
                    if not rel63.isZero: relations.append(rel63)
            return relations
        
        addedRels = []
        processed_triangulations = []
        relations = AddRelation(relations, self.triangulation, original = True)
        processed_triangulations.append(sorted(self.triangulation.diagonals))

        # We use an additional list to keep track of which triangulation have we considered, so to not do duplicates. Repeatedly factorizing stuff turns out to be very expensive
        for i in range(self.l):
            p, q, r, s = i, (i + 1) % self.l, (i + 2) % self.l, (i + 3) % self.l
            fil1 = [a for a in self.triangulation.diagonals if set(a) == set((p,s))]
            if len(fil1) > 0:
                # For loops are expensive, we check if their colors alternate first
                cols = (self.color(p), self.color(q), self.color(r))
                if cols == (b, w, b) or cols == (w,b,w):
                    # Find the element that cooresponds to the triangulation of this quadrilateral and then swap the diagonals
                    triangulation = [(q, s) if set(a) == set((p,r)) else (p, r) if set(a) == set((q,s)) else a for a in self.triangulation.diagonals]
                    if Contains(processed_triangulations, triangulation): continue
                    processed_triangulations.append(sorted(triangulation))

                    relations = AddRelation(relations, Triangulation(self.l, triangulation), original = False)

        # Print current progress
        if verbose:
            print("\n\n Now process the Exchange relations \n\n")
        
            print("Triangulations:")
            print(*processed_triangulations, sep = "\n")

            print("Pre pass:" + str(len(relations)) + " left")
            print(*relations, sep = "\n")
            print("\n\n")

        # Now filter
        # First pass: filter out all duplicates and everything that is not a valid relation
        for relation in relations:
            if Contains(addedRels, relation): continue
            if not Contains(clustervars, relation.equation[0]): continue
            addedRels.append(relation)

        # Print progress after first pass
        if verbose:
            print("First pass:" + str(len(addedRels)) + " left")
            print(*addedRels, sep = "\n")
            print("\n\n")

        # Do second pass: filter out all the from-to swapped duplicates. One of them is not original
        addedRels2 = addedRels
        if len(addedRels) < len(clustervars):
            print(*addedRels)
            raise AssertionError("Too few added relations!")
        elif len(addedRels) > len(clustervars):
            # Second pass: filter out all the from-to swapped duplicates. One of them is not original
            addedRels2 = []
            for relation in addedRels:
                dualRelation = SkeinRelation(relation.ex.equation, relation.fr.equation, relation.to.equation, relation.signature, relation.original)
                if Contains(addedRels2, relation) or Contains(addedRels2, dualRelation): continue

                # If the list after first pass contains both, then add the one that is original, otherwise just add it
                if Contains(addedRels, dualRelation):
                    if relation.original: addedRels2.append(relation)
                    elif dualRelation.original: addedRels2.append(dualRelation)
                    else: raise AssertionError("Both relations are unoriginal!")
                else: 
                    addedRels2.append(relation)
        if len(addedRels2) != len(clustervars):
            print(*addedRels2, sep = "\n")
            raise AssertionError("There are a wrong number of exchange relations!")
        if verbose:
            print(*addedRels2, sep = "\n")
            print("\n\n\n\n\n")
        
        # Do third pass: some exchange relations might be opposite sides so we need to swap the ones whereever necessary.
        # Create a buffer B (queue) to store exchage relations since some of the "arbitrary choices" we make might softlock us later. Free will is a lie
        # Fix the first one by going through all cluster variables and frozen vars
        # Create a list L of processed exchange relation and add the first one in
        # Add all other equations into B
        # Make a counter = 0 to see if the buffer went through one full cycle then that means 
        # while B is not empty:
        #   Get the first equation and see if it can only be one of the orientations
        #   if yes, (swap it and) add it to L; also reset counter to len(B)
        #   otherwise add it back to B and counter -= 1
        #   if counter == 0:
        #       We need to make an arbitrary choice. So we fix the first one from the list.

        # Cleans up the cluster variable for you
        # This passed sanity check
        def FlushCV():
            for cv in clusterVariables:
                to, fr = [], []
                [to.append(x) for x in cv.toTerms if x not in to]
                [fr.append(x) for x in cv.fromTerms if x not in fr]
                cv.toTerms, cv.fromTerms = to, fr

        # Grabs the cluster variable from the list of generated cluster variables. Raises an error otherwise
        # This passed sanity check
        def GetClusterVar(var: J):
            filter = [x for x in clusterVariables if x.j == var] + [x for x in frozenVariables if x.j == var]
            assert len(filter) == 1
            return filter[0]
        
        # Fixes the arrows incident to a cluster variable in the quiver by "registering" th arrows to the quiver
        # This passed sanity check
        def Fix(relation: SkeinRelation):
            if verbose: print("Fixing the equation " + str(relation))
            x = GetClusterVar(relation.ex[0])
            x.toTerms += [GetClusterVar(a) for a in relation.to]
            x.fromTerms += [GetClusterVar(a) for a in relation.fr]
            # Handle from terms and to tems
            for term in relation.fr:
                y = GetClusterVar(term)
                y.toTerms += [x]
            for term in relation.to:
                y = GetClusterVar(term)
                y.fromTerms += [x]
            FlushCV()
        
        # Check if a term is compatible with the equation
        def CheckTerm(equation: SkeinRelation, x: ClusterVariable):
            exitEarly = False
            # For an equation to be compatible, the exchange term needs to have its from (to) terms correctly placed
            # i.e. If A is supposed to be a toTerm in x, then x cannot be a to term in A
            # If it did not exit early that means the equation passed all the tests as is. That's fantastic!
            for toTerm in equation.to:
                if exitEarly: break
                A = GetClusterVar(toTerm)
                filter = [y for y in A.toTerms if x == y]
                assert len(filter) == 0 or len(filter) == 1
                if len(filter) == 1: 
                    exitEarly = True
            for fromTerm in equation.fr:
                if exitEarly: break
                A = GetClusterVar(fromTerm)
                filter = [y for y in A.fromTerms if x == y]
                assert len(filter) == 0 or len(filter) == 1
                if len(filter) == 1: 
                    exitEarly = True
            return not exitEarly

        # Check if this equation is fixed under the rest of the equation. If yes, then return True. If yes after swapped, 
        # then it swaps it for you and returns True. Otherwise return False
        def isFixed(equation: SkeinRelation):
            x = GetClusterVar(equation.ex[0])
            orig = CheckTerm(equation, x)
            equation.swap()
            swap = CheckTerm(equation, x)
            equation.swap()
            # If both sides are ok then its not fixed
            if orig == swap: return False
            if orig: return True
            equation.swap()
            return True

        buffer = [addedRels2[i] for i in range(1, len(addedRels2))]
        processed_relations = [addedRels2[0]]
        counter = len(buffer)
        Fix(addedRels2[0])
        while len(buffer) > 0:
            if verbose: print("Third phase: buffer length is now" + str(len(buffer)))
            eqn = buffer.pop(0)
            if isFixed(eqn):
                Fix(eqn)
                processed_relations += [eqn]
                counter = len(buffer)
            else:
                buffer += [eqn]
                counter -= 1
            if counter == 0 and len(buffer) > 0:
                eqn = buffer.pop(0)
                Fix(eqn)
                processed_relations += [eqn]

        assert len(processed_relations) == len(clusterVariables)

        # Might as well also generate the exchange matrix
        self.cluster = ClusterAlgebra(clusterVariables, frozenVariables, processed_relations)
        self.clusterVariables = clustervars
        self.frozenVariables = frozen
        self.generated = True
        
    # Print all the cluster and frozen variables in a diagram
    def PrintAllClusterVariables(self):
        if not self.generated: self.GenerateCluster()
        for cv in self.clusterVariables:
            dia = TensorDiagram(self.signature)
            dia.AddJInvariant(cv)
            print("This diagram corresponds to the cluster variable \\({}\\)".format(cv))
            print(dia)
        for fv in self.frozenVariables:
            dia = TensorDiagram(self.signature)
            dia.AddJInvariant(fv)
            print("This diagram corresponds to the frozen variable \\({}\\)".format(fv))
            print(dia)

# Generate signature from simplified combinatorial data
def GenerateSignature(t: tuple, length = 9):
    # First take care of the degenerate case
    if t == (): return [b for _ in range(length)]

    sum_t = np.sum(np.array(t,    dtype = int))
    assert len(t) + sum_t == length
    triangulation = []
    for value in t:
        triangulation += [b for _ in range(value)]
        triangulation += [w]
    return triangulation

# Generate comparison diagrams for all cluster variables and frozen variables
def CompareVariable(cv, cv2, sig, sig2, scale=1.5):
        dia = TensorDiagram(sig)
        dia2 = TensorDiagram(sig2)

        dia.AddJInvariant(cv)
        dia2.AddJInvariant(cv2)

        dia.shift = (-1.5, 0)
        dia2.shift = (1.5, 0)

        dia.OptimizeDiagram()
        dia2.OptimizeDiagram()

        # Attempt to print both of them together
        print("This diagram corresponds to the cluster variable \\({}\\) and \\({}\\)".format(cv, cv2))
        st = f"\n\\centering\n\\begin{{tikzpicture}}[scale = {scale}]\n\t\\filldraw[color=black, fill=white!0, thin](1.5,0) circle (1);"
        st += "\t\\filldraw[color=black, fill=white!0, thin](-1.5,0) circle (1);"
        st += dia.__generateStrContents__()
        st += dia2.__generateStrContents__()
        st += "\n\\end{tikzpicture}\n\\par\n\\raggedright\n\n"
        return st

# Generate comparison diagrams for all cluster variables and frozen variables
def CompareCV(diagram1, diagram2, scale=1.5):
    if not diagram1.generated: diagram1.GenerateCluster()
    if not diagram2.generated: diagram2.GenerateCluster()

    # Get the correct n value
    n = len(diagram1.signature)

    # Loop through every cluster variable and then frozen variable
    for i in range(2 * n - 8): print(CompareVariable(diagram1.clusterVariables[i], diagram2.clusterVariables[i], diagram1.signature, diagram2.signature, scale=1.5))
    for i in range(n): print(CompareVariable(diagram1.frozenVariables[i], diagram2.frozenVariables[i], diagram1.signature, diagram2.signature, scale=1.5))

# Gets basic information about the cluster
def GetInfo(diagram1, diagram2 = None, printer = False, print_quiver = False):
    if not diagram1.generated: diagram1.GenerateCluster()
    print(" Cluster: ", end = " ")
    print(*diagram1.clusterVariables, end = " ")
    print(" Frozen: ", end = " ")
    print(*diagram1.frozenVariables)
    if printer: print(*diagram1.cluster.exchangeRelations, sep = "\n")

    if diagram2 is not None:
        if not diagram2.generated: diagram2.GenerateCluster()
        print(" Cluster: ", end = " ")
        print(*diagram2.clusterVariables, end = " ")
        print(" Frozen: ", end = " ")
        print(*diagram2.frozenVariables)
        if printer: print(*diagram2.cluster.exchangeRelations, sep = "\n")
        print(diagram1.clusterVariables == diagram2.clusterVariables)
        
    if print_quiver:
        print(diagram1.quiver)
        if diagram2 is not None: print(diagram2.quiver)

# Number of arrows in a quiver
def weight(B_matrix):
    w = np.count_nonzero(np.abs(B_matrix))
    if w % 2: raise AssertionError
    return w // 2