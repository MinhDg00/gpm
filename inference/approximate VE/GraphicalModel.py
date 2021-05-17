from Factor import Factor
import collections 
import heapq 

class GraphicalModel:

    def __init__(self):
        '''
        Intialize Graphical Model Object
        '''
        self.factors = [] 
        self.cardinalities = []
        self.variables = []
        self.numCliques = 0
        self.evid = {}
        self.minDegreeOrder = []
        self.neighbors = collections.defaultdict(set)
        self.tree_decomposition = collections.defaultdict(set)

    def read(self, inputFile):
        '''
        Read input file and populate information for the graphical model

        Args:
         inputFile: input UAI file 
        ''' 

        with open(inputFile) as fname:

            fname.readline()

            numVars = int(fname.readline())
            for i in range(numVars):
                self.variables.append(i)

            self.cardinalities = list(map(int, fname.readline().split()))

            self.numCliques = int(fname.readline())
            cliques = []
            for i in range(self.numCliques):
                cliques.append(list(map(int, fname.readline().split()))[1:])
            tables = []
            for i in range(self.numCliques):
                while( len(fname.readline()) <= 1): continue
                # fname.readline()
                tables.append(list(map(float, fname.readline().split())))

            for i in range(self.numCliques):                    
                self.factors.append(Factor(cliques[i], tables[i], self.cardinalities))


    def buildNeighbor(self):
        '''
        Build adjancency matrix to store neighbor of all variables
        '''
        for f in self.factors:
            if f.getScope():
                for var in f.getScope():
                    self.neighbors[var] |= set(f.getScope()) 


    def getMinNeighbor(self):
        '''
        Get the next variable in minimum degree
        '''
        # find variable with min degree
        minlen = float('inf')
        minvar = float('inf')
        for v in self.neighbors:
            if len(self.neighbors[v]) < minlen:
                minlen = len(self.neighbors[v])
                minvar = v
            # elif len(self.neighbors[v]) == minlen:
            #     if v < minvar:
            #         minvar = v
        
        self.tree_decomposition[minvar] = self.neighbors[minvar].copy()
        # eliminate variable from neighbors and add edges to all children of variable
        self.neighbors[minvar].remove(minvar)
        if self.neighbors[minvar]:
            for var in self.neighbors[minvar]:
                self.neighbors[var] |= self.neighbors[minvar]
                self.neighbors[var].remove(minvar)

        del self.neighbors[minvar]
        return minvar 

    

    def order(self):
        '''
        Perform min-degree ordering 
        '''
        self.minDegreeOrder = []
        self.buildNeighbor()
        numVar = len(self.neighbors)
        for _ in range(numVar):
            var = self.getMinNeighbor()
            if var not in self.evid:
                self.minDegreeOrder.append(var)

    def instantiateFile(self, evidFile):
        '''
        Instantiate evidence. Reduce CPTs and factors
        '''

        with open(evidFile) as fname:
            lst = list(map(int, fname.readline().split()))
            for i in range(1, len(lst), 2):
                self.evid[lst[i]] = lst[i+1]

        for i in range(len(self.factors)):
            for v in self.evid: 
                if self.factors[i].contains(v):
                    s = self.factors[i].getStride(v)

                    tmpScope = self.factors[i].getScope()
                    tmpScope.remove(v)
                    
                    tmpTable = []
                    for j in range(self.evid[v] * s, self.factors[i].getTableSize(), s * self.cardinalities[v]):
                        for k in range(j, j + s):
                            tmpTable.append(self.factors[i].getTable()[k])
                        
                    self.factors[i] = Factor(tmpScope, tmpTable, self.cardinalities)
    
    def instantiateSample(self, evidSample):
        '''
        Instantiate evidence. Reduce CPTs and factors
        '''
        for var, val in evidSample.items():
            self.evid[var] = val 
        for i in range(len(self.factors)):
            for v in self.evid: 
                if self.factors[i].contains(v):
                    s = self.factors[i].getStride(v)

                    tmpScope = self.factors[i].getScope()
                    tmpScope.remove(v)
                    
                    tmpTable = []
                    for j in range(self.evid[v] * s, self.factors[i].getTableSize(), s * self.cardinalities[v]):
                        for k in range(j, j + s):
                            tmpTable.append(self.factors[i].getTable()[k])
                        
                    self.factors[i] = Factor(tmpScope, tmpTable, self.cardinalities)



    def product(self, f1, f2):
        '''
        Return product of 2 factors. Code reference from Box 10.A, pages 358-361 in Koller and Friedman
        Args:
         f1, f2: Factor() objects
        '''
        j, k = 0,0 
        assignment = {}
        tableSize = f1.getTableSize() 

        for var in f1.getScope():
            assignment[var] = 0

        for var in f2.getScope():
            if not f1.contains(var):
                tableSize *= self.cardinalities[var]
                assignment[var] = 0

        newTable = [0] * tableSize

        for i in range(tableSize):
            newTable[i] = f1.getTable()[j] * f2.getTable()[k]
            for l in reversed(sorted(assignment.keys())):
                assignment[l] += 1
                if assignment[l] == self.cardinalities[l]:
                    assignment[l] = 0  
                    j -= (self.cardinalities[l] - 1) * f1.getStride(l)
                    k -= (self.cardinalities[l] - 1) * f2.getStride(l)
                else:
                    j += f1.getStride(l)
                    k += f2.getStride(l)
                    break
        
        return Factor(list(sorted(assignment.keys())), newTable, self.cardinalities)

    def sumout(self, f, var):
        '''
        Return new factor as marginalization of f on var
        Args:
         f: Factor() object
         var: variable in Markov Network
        '''
        tableSize = f.getTableSize()//self.cardinalities[var]
        newTable = [0] * tableSize
        newScope = f.getScope()
        newScope.remove(var)
        s = f.getStride(var)
        seen = set()
        assign = 0

        j = 0
        for i in range(tableSize):
            while True:
                while j in seen:
                    j += 1
                seen.add(j)
                newTable[i] += f.getTable()[j]
                assign += 1
                if assign == self.cardinalities[var]:
                    assign = 0
                    j -= (self.cardinalities[var] - 1) * s
                    break 
                else:
                    j += s
        
        return Factor(newScope, newTable, self.cardinalities)
        
    def getOrder(self):
        return self.minDegreeOrder



