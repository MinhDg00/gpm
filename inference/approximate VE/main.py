from Factor import Factor
from GraphicalModel import GraphicalModel
import sys 
import math
from helper import SVEC
import random 
import time 


inputFile = sys.argv[1]
evidFile = sys.argv[2]

while True:
    
    w = int(input('Please enter w-cutset bound integer: '))
    N = int(input('Please enter number of samples you want to use: '))
    Q_type = input('Input proposal distribution option you want to use (adaptive/uniform): ')

    # Read network and instatiate model with evidence  
    G = GraphicalModel()
    G.read(inputFile)
    G.instantiateFile(evidFile)

    s = time.time() 
    res = SVEC(G, w, N, Q_type)

    print('The probability of evidence or partion function using {} proposal distribution is: {}'.format(Q_type, res))
    print('It take {}s to run'.format(time.time() - s))
    
    cont = input('Do you want to continue (y/n): ')

    if cont.lower() == 'n':
        break