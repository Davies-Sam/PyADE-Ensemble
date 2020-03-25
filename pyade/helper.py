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



############################################
#                 GLOBALS                  #
############################################
RUNS = 1
runsDict = {}
mediansDict = {}

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
    problems.Ackley().evaluate : {
                "name" : "Ackley",
                "bounds" : problems.Ackley, 
                "graph" : problems.Ackley().plot3d,
                "answer" : 0
                },
    problems.Rosenbrock().evaluate : {
            "name" : "Rosenbrock's Saddle",
            "bounds" : problems.Rosenbrock, 
            "graph" : problems.Rosenbrock().plot3d,
            "answer" : 0
             },
    problems.EggHolder().evaluate: { #2d
            "name" : "Eggholder",
            "bounds" : problems.EggHolder, 
            "graph" : problems.EggHolder().plot3d,
            "answer" : -959.6407
             },
    problems.Rastrigin().evaluate : {
            "name" : "Rastigrin",
            "bounds" : problems.Rastrigin, 
            "graph" : problems.Rastrigin().plot3d,
            "answer" : 0
             },
    problems.Schwefel().evaluate : {
            "name" : "Schwefel",
            "bounds" : problems.Schwefel, 
            "graph" : problems.Schwefel().plot3d,
            "answer" : 0
             },
    problems.Easom().evaluate : { #2d
            "name" : "Easom",
            "bounds" : problems.Easom, 
            "graph" : problems.Easom().plot3d,
            "answer" : -1
             },
    problems.Levy().evaluate : {
            "name" : "Levy",
            "bounds" : problems.Levy, 
            "graph" : problems.Levy().plot3d,
            "answer" : 0
             },
    problems.Michalewicz().evaluate : {
            "name" : "Michalewicz",
            "bounds" : problems.Michalewicz, 
            "graph" : problems.Michalewicz().plot3d,
            "answer" : -1.8013
             },
    problems.StyblinskiTang().evaluate : {
            "name" : "StyblinskiTang", 
            "bounds" : problems.StyblinskiTang, 
            "graph" : problems.StyblinskiTang().plot3d,
            "answer" : -39.16599 * 2
             },
    problems.CrossInTray().evaluate : {
            "name" : "CrossInTray",
            "bounds" : problems.CrossInTray, 
            "graph" : problems.CrossInTray().plot3d,
            "answer" : -2.06261
             },
    problems.DixonPrice().evaluate : {
            "name" : "DixonPrice",
            "bounds" : problems.DixonPrice, 
            "graph" : problems.DixonPrice().plot3d,
            "answer" : 0
             },
    problems.HolderTable().evaluate : {
            "name" : "HolderTable",
            "bounds" : problems.HolderTable, 
            "graph" : problems.HolderTable().plot3d,
            "answer" : -19.2085
             }
    }

############################################
#         MATPLOTLIB PLOTTING              #
############################################

def updateRuns(function: Callable, algo: ModuleType, x: int, result: Union[float, int]):
        """Updates the runs dictionary"""
        runsDict[(function, algo, x)] = result

def storeMedianResult(function: Callable, algo: ModuleType ):
        """Calculates the median fitness per generation for an algorithm's performance over a function using runsDict and then updates mediansDict"""
        genlimit = [len(runsDict[key]) for key in runsDict.keys()]
        limitIndex = np.argmax( genlimit )
        limit = genlimit[limitIndex]
        generations = [ [] for i in range(limit) ]

        #each key is 1 run
        runs = [key for key in runsDict.keys() if function in key if algo in key]

        for key in runs:
                for x in range(0, len(runsDict[key])):
                #print('run %s - gen %s ||' % (key[2], x), runsDict[key][x][1] )
                # for each run (key), get gen 'x' best fitness
                        generations[x].append(runsDict[key][x][1])

        #create and store median for each array of generation values
        medians = []
        for gen in generations:
                medians.append( np.median(gen) )
        #store the medians array with corresponding algo and function
        mediansDict[ (algos[algo], functions[function]['name']) ] = medians
        animate2D(function,algo)
        plt.clf()


def plotMedians(function: Callable):
        """Creates a single matplot graph comparing all algorithms over a function"""
        test = [key for key in mediansDict.keys() if functions[function]['name'] in key]
        plt.clf()

        for key in test:
                #choice = np.random.choice(options)
                values = mediansDict[key]
                #vals = [val for val in values if val != functions[function]['answer']]
                plt.plot(values, label='%s ' % key[0], alpha=0.9)
                #options.remove(choice)
        plt.legend(loc='upper center', bbox_to_anchor=(.95, 1),
                ncol=1, fancybox=True, shadow=True)
 
        ## negative plots look bad, fix this, add animations with the graphs.
        if functions[function]['answer'] < 0:
                plt.yscale('linear')
                
        else:
                plt.yscale('log')

        plt.autoscale()
        plt.xlabel('$Generations$')
     
        funcName = functions[function]['name']
        name = '$%s$  - median of  %s  runs' % (funcName,  RUNS)
        plt.title(name)

        newpath = 'Graphs2/%s' % (funcName)
        if not os.path.exists(newpath):
                os.makedirs(newpath)
        plt.savefig('Graphs2/%s/%s.png' % (funcName, name) )
        plt.show()    
        plt.clf()
        
############################################
#               Animations                 #
############################################

def animate3D(function: Callable, algo: ModuleType):
        """Creates a 3D animation of one algo on one plot"""
        results = runsDict[(function,algo, 0)]
        fig = plt.figure()
        camera = Camera(fig)

        for gen in results:
                x = [x1[0] for x1 in gen[2]]
                y = [x2[1] for x2 in gen[2]]
                z = gen[3]
                
                ax = fig.add_subplot(111, projection='3d')
                test = functions[function]['graph'](ax3d=ax)
                test.scatter(x,y,z)
                
                camera.snap()
                #plt.show()
                #exit()
        funcName = functions[function]['name']
        newpath = '/Graphs/%s' % (funcName)
        if not os.path.exists(newpath):
                os.makedirs(newpath)

        animation = camera.animate()
        name = funcName + '' + algos[algo]
        animation.save('/Graphs/%s/3D%s.mp4' % (funcName, name))
        plt.close()

        #######################
        # this is how to scatter solutions over the 3d graph to make animations
        #fig = plt.figure(figsize=(12,8))
        #ax = fig.add_subplot(111, projection='3d')
        #test = functions[function]['graph'](ax3d=ax)
        #test.scatter([1,2,3], [1,2,3], [1,2,3])
        #plt.show()   
    
def animate2D(function, algo):
        """Creates a 2D animation of one algo on one plot"""
        results = runsDict[(function,algo, 0)]
        fig = plt.figure(figsize=(12,8))
        camera = Camera(fig)

        for i, gen in enumerate(results):
                x = [x1[0] for x1 in gen[2]]
                y = [x2[1] for x2 in gen[2]]
                fig, ax =functions[function]['bounds']().plot2d(figure=fig)
                ax.scatter(x,y)
                ax.text(0.5, 1.01, 'Generation %s' % i, transform=ax.transAxes)
                camera.snap()
                #plt.show()

        funcName = functions[function]['name']
        newpath = 'Graphs2/%s' % (funcName)
        if not os.path.exists(newpath):
                os.makedirs(newpath)

        animation = camera.animate()
        name = funcName + '' +algos[algo]

        try:
                animation.save('Graphs2/%s/2D%s.mp4' % (funcName, name))
        except subprocess.CalledProcessError:
                pass
        plt.close()

def animate2D_all(function, algo):
        """Creates a 2D animation of all algos on one plot"""
        results = runsDict[(function,algo, 0)]
        fig = plt.figure(figsize=(12,8))
        camera = Camera(fig)

        for i, gen in enumerate(results):
                x = [x1[0] for x1 in gen[2]]
                y = [x2[1] for x2 in gen[2]]
                fig, ax =functions[function]['bounds']().plot2d(figure=fig)
                ax.scatter(x,y)
                ax.text(0.5, 1.01, 'Generation %s' % i, transform=ax.transAxes)
                camera.snap()
                #plt.show()

        funcName = functions[function]['name']
        newpath = 'Graphs/%s' % (funcName)
        if not os.path.exists(newpath):
                os.makedirs(newpath)

        animation = camera.animate()
        name = funcName + '' +algos[algo]
        animation.save('/Graphs/%s/2D%s.mp4' % (funcName, name), writer='imagemagick' )
        plt.close()


