import numpy as np
import commons
from helper import  de, jade, sade, shade
import os
K = 3
DIM = 0

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

# pop size is the num of evals 
def apply(population_size: int, individual_size: int, bounds: np.ndarray,
        func, opts,
        callback,
        max_evals: int, 
        seed,
        population,
        answer ) :

    ensemble = [de, sade, jade, shade]

    algoParams = {}


    n = int(population_size / len(ensemble))
    
    np.random.shuffle(population)
    #print(population)
    splitPopulation = [population[i:i + n] for i in range(0, len(population), n)]
    #print(splitPopulation)
    for i, algo in enumerate(ensemble):
        params = algo.get_default_params(dim=DIM)
        bounds = np.array(bounds * DIM)
        params['func'] = func
        params['bounds'] = bounds
        params['opts'] = None
        params['answer'] = None
        params['population'] = splitPopulation[i].copy()
        params['population_size'] = len(params['population'])
        algoParams[algo] = params

    finalResult = []
    evals = 0
    numGens = 0

    while evals < max_evals:
        results = {de : [], sade : [], jade : [], shade : []}  
        ensemble = [algo for algo in ensemble if algoParams[algo]['population_size'] > 0]
        ########### shuffle the population amongst the variants again. ##############
        
        if len(ensemble) > 1:
            totalPop = np.concatenate([algoParams[algo]['population'] for algo in ensemble])
            np.random.shuffle(totalPop)
            #print('total pop size before splitting: %s' % len(totalPop))
            for algo in ensemble:
                #print('algo %s, pop %s' % (algo, algoParams[algo]['population']))
                #print('algo %s, size %s, old pop: %s' % (algo, algoParams[algo]['population_size'], algoParams[algo]['population']))
                size = algoParams[algo]['population_size']    
                algoParams[algo]['population'] = totalPop[:size]
                totalPop = totalPop[size:]
                #print('algo %s, pop %s' % (algo, algoParams[algo]['population']))
                #print('total pop size in splitting : %s' % len(totalPop))
                #print('algo %s, size %s, new pop: %s' % (algo, algoParams[algo]['population_size'], algoParams[algo]['population']))


        for i in range(0, K):    
            #print(ensemble)
            for algo in ensemble:
                params = algoParams[algo] 
                popSize = algoParams[algo]['population_size']
                #params['population'].flags.writeable = True
                params['population_size'] = popSize
                params['max_evals'] = popSize
                algoParams[algo] = params
                
                if algo == jade:
                    params['p'] = max( .05, 3 / params['population_size'] )
                result = algo.apply(**params)
                  
                for res in result:
                    results[algo].append(res)
                # print(len(results[algo]))        
                algoParams[algo]['population'] = results[algo][-1][2].copy()
                #print(type(algoParams[algo]['population']))
                evals += popSize
            numGens += 1

            if evals == max_evals:
                break
 
        fitnesses = {}
        fitnesses.clear()
        bestFitnesses = {}
        bestFitnesses.clear()
        fitnessHistory = {}
        #print(ensemble)
        if len(ensemble) > 1:
            for algo in ensemble:
                if algoParams[algo]['population_size'] != 0:
                    algoParams[algo]['population'] = results[algo][-1][2].copy()
                    fitnesses[algo] = results[algo][-1][3].copy()
                    fitnessHistory[algo] =  [ results[algo][x][3] for x, gen in enumerate(results[algo]) ]
                    bestFitnesses[algo] = results[algo][-1][1].copy()

            scores = [ (bestFitnesses[algo], algo) for algo in bestFitnesses.keys()]

            test = [ (fitnessHistory[algo][0] - fitnessHistory[algo][-1]) / (algoParams[algo]['population_size'] * K)  for algo in fitnessHistory.keys()] 
            
            #implement ratio of improvement over evals
            #print(test)
            #newScores = max ( test )
            #print('new :', newScores)


            scores.sort(key=lambda x: x[0])


            #print("scores : ", scores)

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
                if len(worstList) != 0:
                    bestList.append( worstList.pop() )

            assert oldLenW - len(worstList) == len(bestList) - oldLenB , "migration error"
           

            bestPopulation = np.array([item[0] for item in bestList])
            worstPopulation = np.array([item[0] for item in worstList])

            algoParams[best]['population'] = bestPopulation.copy()
            algoParams[worst]['population'] = worstPopulation.copy()  
            algoParams[best]['population_size'] = len(algoParams[best]['population'])
            algoParams[worst]['population_size'] = len(algoParams[worst]['population'])

            bestOrg = np.array( [gen[0].copy() for gen in results[best] ] )
            bestFit = np.array( [gen[1].copy() for gen in results[best] ] )

            pop = np.array( [ gen[2].copy() for gen in results[algo]  for algo in ensemble]  ) 
            fit = np.array( [ gen[3].copy() for gen in results[algo]  for algo in ensemble] ) 

            for i, gen in enumerate(results[best]):
                finalResult.append( (bestOrg[i], bestFit[i], pop, fit) )

        elif len(ensemble) == 1:
            bestOrg = np.array( [gen[0].copy() for gen in results[ensemble[0]] ] )
            bestFit = np.array( [gen[1].copy() for gen in results[ensemble[0]] ] )

            pop = np.array( [gen[2].copy() for gen in results[ensemble[0]]] ) 
            fit = np.array( [gen[3].copy() for gen in results[ensemble[0]] ] ) 

            for i, gen in enumerate(results[ensemble[0]]):
                finalResult.append( (bestOrg[i], bestFit[i], pop, fit) )
    ####print('numGens: ', numGens)
    return finalResult
        #add dict to track algo pop growth
    