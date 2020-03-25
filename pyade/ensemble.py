# Import all the DE algorithm variants from python Advanced DE libarary
import numpy as np
from helper import functions, algos, updateRuns, plotMedians, storeMedianResult, RUNS
import os



############################################
#             Directory Setup              #
############################################
local = os.getcwd()
dirName = local + '\\Graphs2' 
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

#Initialize a dictionary to store 50 runs & the generation number for each algorithm 
for i, function in enumerate(functions.keys()):
    #startingPopulations = [ commons.init_population(10 * dimensions, dimensions, np.array( functions[function]['bounds']().bounds )) for x in range(runs)]
    for j, algo in enumerate(algos.keys()):
        for x in range(0, RUNS): 
            # Set the parameters to the default for a problem with 2 variables
            params = algo.get_default_params(dim=2)
            # Define the boundaries of the variables   
            a = functions[function]['bounds']().bounds   
            bounds = np.array(a)           
            params['bounds'] = bounds
            # We indicate the function we want to minimize
            params['func'] = function
            params['opts'] = None
            params['answer'] = functions[function]['answer']
            params['max_evals'] = 5000
            #params['population'] = startingPopulations[x]
            params['population'] = None

            # We run the algorithm and obtain the results
            algorithm = algo.apply(**params)
            result =  list(algorithm)
            updateRuns(function, algo, x, result)

        #Store algorithms median best over 50 runs    
        storeMedianResult(function, algo)
        
    #######################
    # plot medians over each function after each algorithm has 50 runs completed
    plotMedians(function)
