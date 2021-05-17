
class Factor:

    def __init__(self, scope = [], table = [], cardinalities = []):
        '''
        Initializes factor class 

        Args:
         scope: scope of factor
         table: Prob table of all 
         cardinalities: cardinalitites of ALL variables in Markov network
        '''
        self.scope = scope
        self.table = table
        self.cardinalities = cardinalities
        self.strides = {}                                    # stride for variable assignment in array entry
        self.indexTable = {}                                 # keep track of variable index in the scope

        for i, var in enumerate(self.scope):
            self.indexTable[var] = i

        product = self.getTableSize()

        for var in self.scope:
            product //= self.cardinalities[var]
            self.strides[var] = product

    def getVarIndex(self, var):
        '''
        get index of variable in factor scope array
        '''
        if self.contains(var):
            return self.indexTable[var]
        
        return -1 

    def getTableSize(self):
        '''
        Get size of probabiity table
        '''
        return len(self.table)

    def getScopeSize(self):
        '''
        Get size of factor scope
        '''
        return len(self.scope)
    
    
    def getStride(self, var):
        '''
        Get stride of variables in a factor 
        '''        
        if self.contains(var):
            return self.strides[var]

        return 0

    def contains(self, var):
        '''
        Check whether factor contains a variable
        '''
        return False if var not in self.indexTable else True

    def getScope(self):
        '''
        Get scope of a factor 
        '''
        return self.scope

    def getTable(self):
        '''
        Get probability table of a factor
        '''
        return self.table 
    

