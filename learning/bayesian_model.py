import numpy as np 
import random 

class BN:

    def __init__(self):
        '''
        Intialize Bayesian Model 
        '''
        self.CPTs = {}
        self.cardinalities = []
        self.numVars = 0
        self.numCPTs = 0
        self.parents = {}
        self.children = []

    def readModel(self, modelFile):
        '''
        Read model file and populate information for the input Bayesian network

        Args:
         inputFile: input UAI file 
        ''' 

        with open(modelFile) as fname:
            
            fname.readline() 

            self.numVars = int(fname.readline())

            for i in range(self.numVars):
                self.children.append(i)

            self.cardinalities = list(map(int, fname.readline().split()))
            self.numCPTs = int(fname.readline())

            for i in range(self.numCPTs):
                par = list(map(int, fname.readline().split()))[1:]
                child = par.pop() 
                self.parents[child] = par

            tables = [] 
            for i in range(self.numCPTs):
                fname.readline()
                noAssign = int(fname.readline()[:-1])
                tbl = []
                tmp = []
                while True:
                    tmp += list(map(float, fname.readline().lstrip().split()))
                    if len(tmp) == noAssign:
                        break 

                # for _ in range(noAssign//self.cardinalities[self.children[i]]):
                #     tbl.append(np.array(list(map(float, fname.readline().lstrip().split()))))
                for i in range(0, len(tmp), 2):
                    tbl.append(tmp[i:i+2]) 

                tables.append(tbl)
        
            for i in range(self.numCPTs):
                self.CPTs[self.children[i]] = CPT(self.parents[self.children[i]], self.children[i], np.array(tables[i]))

    def randomBN(self, n, seed_number):
        '''
        Create a random singly-connected Bayesian tree network

        n: number of variables 
        '''

        random.seed(seed_number)
        self.numVars = n 
        self.cardinalities = [2] * n
        self.numCPTs = n 
        self.children = [i for i in range(n)]
        variables = [i for i in range(n)]
        random.shuffle(variables)

        for i in range(n):
            if i == 0:
                self.parents[variables[i]] = [] 
            else:
                self.parents[variables[i]] = [variables[i-1]]

        for c in variables:
            parent = self.parents[c]
            rowSize = 2 ** len(parent)

            tbl = []
            for r in range(rowSize):
                c0 = random.random()
                tbl.append([c0, 1-c0])
            
            self.CPTs[c] = CPT(parent, c, np.array(tbl))

class CPT: 

    def __init__(self, parent, child, table):
        
        self.parent = parent
        self.child = child 
        self.table = table

    def getTable(self):
        '''
        Get probability table of a factor
        '''
        return self.table 
        

    def normalize(self):
        '''
        normalize CPT
        '''        
        row_sum = self.table.sum(axis = 1, keepdims = True)
        self.table = np.divide(self.table, row_sum)
        
    def getParent(self):
        return self.parent

    def getChild(self):
        return self.child  

    
    
         
        
        
        




            

