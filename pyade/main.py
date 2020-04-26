# Import all the DE algorithm variants from python Advanced DE libarary
import numpy as np
from helper import functions, algos, updateRuns, plotMedians, storeMeanResult, RUNS
import os
import commons
from cec2005real.cec2005 import Function
############################################
#             Directory Setup              #
############################################
local = os.getcwd()
dirName = local + '/ensembleGraphs' 
try:
    # Create target Directory
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ") 
except FileExistsError:
    pass
    print("Directory " , dirName ,  " already exists")


############################################
#              Main Function               #
############################################
dims = [2, 10, 30]
for dim in dims:
    for funcNum in functions.keys():
        fbench = Function(funcNum, dim)
        info = fbench.info()
        function = fbench.get_eval_function()
        bounds = [(info['lower'], info['upper'])]
        startingPopulations = [ commons.init_population(10 * dim, dim, np.array( bounds )) for x in range(RUNS)]
        for j, algo in enumerate(algos.keys()):
            for x in range(0, RUNS):
                params = algo.get_default_params(dim=dim)
                bounds = np.array(bounds * dim)
                params['func'] = function
                params['bounds'] = bounds
                #params['max_evals'] = 10000
                params['opts'] = None
                params['answer'] = None
                params['population'] = startingPopulations[x].copy()
                result = algo.apply(**params)
                """
                print(type(result), algo)
                print(result[-1])
                print(result[-1][3])
                print(algo, len(result))
                """
                updateRuns(funcNum, algo, x, result)
       
            storeMeanResult(funcNum, algo)

        plotMedians(funcNum, dim)