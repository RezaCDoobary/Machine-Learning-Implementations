import os, sys
sys.path.insert(0, os.getcwd() + str("//ML")) 
sys.path.insert(0, os.getcwd())  

from linearModels import *
from measurements import *
from mixtureModels import *
from optimiser import *
from preprocessing import *
from linearClassifierModels import *
from discriminantModels import *
from kernels import *   
from naiveBayes import *
from tree import *