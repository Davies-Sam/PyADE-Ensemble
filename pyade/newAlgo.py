





for j, algo in enumerate(algos.keys()):
    for x in range(0, RUNS):
        params = algo.get_default_params(dim=dim)
        bounds = np.array(bounds * dim)
        params['func'] = function
        params['bounds'] = bounds
        params['opts'] = None
        params['answer'] = None
        params['population'] = startingPopulations[x].copy()
        algorithm = algo.apply(**params)
        result =  list(algorithm)
