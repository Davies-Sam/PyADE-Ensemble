from typing import Callable, Union, Dict, Any
from types import ModuleType
import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib.pyplot as plt
from math import e 
import matplotlib.gridspec as gridspec
from celluloid import Camera
import commons, de, jade, sade, ilshade, shade, jso, lshade, lshadecnepsin, mpede, saepsdemmts
import yabox.problems as problems
import landscapes.single_objective as landscapes
from cycler import cycler
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from IPython.display import HTML
from matplotlib.animation import FFMpegWriter
import os
import subprocess
from cec2019comp100digit import cec2019comp100digit
import numpy

############################################
#                 GLOBALS                  #
############################################
RUNS = 100
runsDict = {}
medianBestFitnessDict = {}
medianMeanFitnessDict = {}


############################################
#       DE Variants & Test Functions       #
############################################

algos = {
    de : 'de',
    sade : "sade",
    jade :"jade",
    shade : "shade",
    #lshade : "lshade", 
    #ilshade : "ilshade", 
    #jso : "jso", 
    #lshadecnepsin : "lshadecnepsin", 
    #mpede : "mpede",
    #saepsdemmts : "saepsdemmts" 
}


functions = {
         1 : {
                "name" : "Shifted Sphere Function",          
         },   
         2 : {
                "name" : "Shifted Schwefel's Problem 1.2",          
         },
         3 : {
                "name" : "Shifted Rotated High Conditioned Elliptic Function",      
         },
         4 : {
                "name" : "Shifted Schwefel's Problem 1.2 with Noise in Fitness",           
         },
         5 : {
                "name" : "Schwefel's Problem 2.6 with Global Optimum on Bounds",     
         },
         6 : {
                "name" : "Shifted Rosenbrock's Function",       
         },
         7 : {
                "name" : "Shifted Rotated Griewnk's Function without Bounds",        
         },
         8 : {
                "name" : "Shifted Rotated Ackley's Function with Global Optimum on Bounds",      
         },
         9 : {
                "name" : "Shifted Rastrigin's Function",          
         },
         10 : {
                "name" : "Shifted Rotated Rastrigin's Function",         
         },
         11 : {
                "name" : "Shifted Rotated Weierstrass Function",        
         },
         12 : {
                "name" : "Schwefel's Problem 2.13",         
         }
}

############################################
#         MATPLOTLIB PLOTTING              #
############################################

def updateRuns(funcNum: int, algo: ModuleType, x: int, result: Union[float, int]):
       """Updates the runs dictionary"""
       runsDict[(funcNum, algo, x)] = result

def storeMeanResult(funcNum: int, algo: ModuleType ):
       """Calculates the median fitness per generation for an algorithm's performance over a function using runsDict and then updates mediansDict"""
       genlimit = [len(runsDict[key]) for key in runsDict.keys()]
       limitIndex = np.argmax( genlimit )
       limit = genlimit[limitIndex]
       generationsBestFitness = [ [] for i in range(limit) ]
       generationsMeanFitness = [ [] for i in range(limit) ]

       #each key is 1 run
       runs = [key for key in runsDict.keys() if funcNum in key if algo in key]
       #for every run
       for key in runs:
              #for every generation 
              for x in range(0, len(runsDict[key])):
                     #print("\nbest vector : ", runsDict[key][x][0])
                     #print("\nbest fitness : ", runsDict[key][x][1])
                     #print("\nall vectors : ", runsDict[key][x][2])
                     #print("\nall fitness : ", numpy.sort(runsDict[key][x][3]))         
                     generationsBestFitness[x].append(runsDict[key][x][1])
                     generationsMeanFitness[x].append(np.mean(runsDict[key][x][3]))
                     
       #create and store median for each array of generation values
       mediansBestFitArr = []
       mediansMeanFitArr = []

       for gen in generationsBestFitness:
              mediansBestFitArr.append( np.median(gen) )
       for gen in generationsMeanFitness:
              mediansMeanFitArr.append( np.median(gen))

       medianBestFitnessDict[ (algos[algo], functions[funcNum]['name']) ] = mediansBestFitArr
       medianMeanFitnessDict[ (algos[algo], functions[funcNum]['name']) ] = mediansMeanFitArr
      
       plt.clf()

def plotMedians(funcNum: int, dim: int):
       """Creates a single matplot graph comparing all algorithms over a function"""
       mBestFitness = [key for key in medianBestFitnessDict.keys() if functions[funcNum]['name'] in key]
       plt.clf()

       for key in mBestFitness:
              values = medianBestFitnessDict[key]
              answerAt = 'Not Found'
              for i, value in enumerate(values):
                     if value == 0:
                            answerAt = i
                            break
              plt.plot(values, label='%s-%s' % (key[0], answerAt), alpha=0.9)
       plt.legend(loc='upper center', bbox_to_anchor=(.95, 1),
              ncol=1, fancybox=True, shadow=True)

       plt.yscale('symlog')
       plt.autoscale()
       plt.xlabel('$Generations$')

       funcName = functions[funcNum]['name']
       name = '$%s$  - D = %s, median-best, %s runs' % (funcName, dim, RUNS)
       plt.title(name)

       newpath = 'GraphsCEC2005/%s' % (funcName)
       if not os.path.exists(newpath):
              os.makedirs(newpath)
       plt.savefig('%s/%s.png' % (newpath, name) )
       plt.show()    
       plt.clf()

       mMeanFitness = [key for key in medianMeanFitnessDict.keys() if functions[funcNum]['name'] in key]
       plt.clf()

       ##### mean fitness graph ######
       for key in mMeanFitness:
              values = medianMeanFitnessDict[key]
              plt.plot(values, label='%s' % key[0], alpha=0.9)
       plt.legend(loc='upper center', bbox_to_anchor=(.95, 1),
              ncol=1, fancybox=True, shadow=True)

       plt.yscale('symlog')
       plt.autoscale()
       plt.xlabel('$Generations$')

       funcName = functions[funcNum]['name']
       name = '$%s$-D=%s, median-mean, %sruns' % (funcName, dim, RUNS)
       plt.title(name)

       newpath = 'GraphsCEC2005/%s' % (funcName)
       if not os.path.exists(newpath):
              os.makedirs(newpath)
       plt.savefig('%s/%s.png' % (newpath, name) )
       plt.show()    
       plt.clf()
       
