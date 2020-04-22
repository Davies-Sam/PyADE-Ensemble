import numpy as np
import commons
import ray
from concurrent.futures import ProcessPoolExecutor as Executor
import concurrent.futures
from helper import  de, jade, sade, shade

K = 5
DIM = 0
ensemble = [de, sade, jade, shade]

def get_default_params(dim: int) -> dict:
    """
        Returns the default parameters of the JADE Differential Evolution Algorithm.
        :param dim: Size of the problem (or individual).
        :type dim: int
        :return: Dict with the default parameters of the JADE Differential
        Evolution Algorithm.
        :rtype dict
        """
    global DIM
    DIM = dim
    return {'callback': None, 'max_evals': 10000 * dim, 'seed': None,
            'individual_size': dim, 'population_size': 10 * dim, 'opts': None}


def apply(population_size: int, individual_size: int, bounds: np.ndarray,
        func, opts,
        callback,
        max_evals: int, 
        seed,
        population,
        answer ) :
    
    #maxEvals = int(max_evals / len(ensemble))
    maxEvals = int( max_evals / K )
    #maxEvals = max_evals

    algoParams = {}
    algoBestSeen = {}
    for algo in ensemble:
        params = algo.get_default_params(dim=DIM)
        bounds = np.array(bounds * DIM)
        params['func'] = func
        params['bounds'] = bounds
        params['opts'] = None
        params['answer'] = None
        params['max_evals'] = maxEvals
        params['population'] = population.copy()
        algoParams[algo] = params

    finalResult = []

    for i in range(0, K):
        results = {}
        results.clear()

        
        bucket_sizes = [ len(algoParams[algo]['population']) for algo in ensemble]
        
        #print(bucket_sizes)
        N = maxEvals
        p = [entry  / float(sum(bucket_sizes)) for entry in bucket_sizes ]
        #print(p)
        limits = np.random.multinomial(N, p)
        limits = [int(limit) for limit in limits]
        #make integer limits.
        #print(limits)
        evalLimitDict = {de: limits[0], jade: limits[1], sade: limits[2] , shade: limits[3] }

        for algo in ensemble:
            params = algoParams[algo] 
            params['population'] = algoParams[algo]['population'].copy()
            params['population'].flags.writeable = True
            params['population_size'] = len(algoParams[algo]['population'])
            params['max_evals'] = evalLimitDict[algo]
            if algo == jade:
                params['p'] = max( .05, 3 / params['population_size'] )
            algoParams[algo] = params
            result = algo.apply(**params)
            results[algo] = result

        #update every algo's population accordingly.
        fitnesses = {}
        bestFitnesses = {}

        for algo in ensemble:
            algoParams[algo]['population'] = results[algo][-1][2].copy()
            fitnesses[algo] = results[algo][-1][3].copy()
            bestFitnesses[algo] = results[algo][-1][1].copy()
            algoBestSeen[algo] = bestFitnesses[algo]

        scores = [ (bestFitnesses[algo], algo) for algo in ensemble]
        scores.sort(key=lambda x: x[0])

        best = scores[0][1]
        worst = scores[-1][1]

        bestPopulation = algoParams[best]['population'].copy()
        worstPopulation = algoParams[worst]['population'].copy()

        bestFitnesses = fitnesses[best]
        worstFitnesses = fitnesses[worst]
        #bestIndex = np.argmin(bestFitnesses)
        bestList = []
        worstList = []

        for j, org in enumerate(bestPopulation):
            bestList.append( (bestPopulation[j], bestFitnesses[j]) )
        for k, org in enumerate(worstPopulation):
            worstList.append( (worstPopulation[k], worstFitnesses[k]) )

        bestList.sort(key=lambda  x: x[1])
        worstList.sort(key=lambda x: x[1])

        oldLenB = len(bestList)
        oldLenW = len(worstList)
    
        for x in range(0, 2):
            bestList.append( worstList.pop() )

        assert oldLenW - len(worstList) % 2 != 0, "migration error"
        assert len(bestList) - oldLenB % 2 != 0, "migration error"

        bestPopulation = np.array([item[0] for item in bestList])
        worstPopulation = np.array([item[0] for item in worstList])

        algoParams[best]['population'] = bestPopulation.copy()
        algoParams[worst]['population'] = worstPopulation.copy()
        

        bestOrg = np.array( [gen[0] for gen in results[best] ] )
        bestFit = np.array( [gen[1] for gen in results[best] ] )

        pop = np.array( [gen[2] for gen in results[best]] ) 
        fit = np.array( [gen[3] for gen in results[best] ] ) 

        for i, gen in enumerate(results[best]):
            finalResult.append( (bestOrg[i], bestFit[i], pop, fit) )
    
    
    return finalResult
    #add dict to track algo pop growth
    