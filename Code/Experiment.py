from numpy.random import seed
from E_CVRP_TW import  E_CVRP_TW, Constructive, Genetic, Feasibility, Reparator, Experiment
from time import time

'''
General parameters
'''
start: float = time()
max_time: int = 3600

rd_seed: int = 0
seed(rd_seed)

path: str = '/Users/juanbeta/My Drive/Research/Energy/CG-VRP-TW/' ##### CHANGE IF NECESSARY!!!


'''
Environment
'''
env: E_CVRP_TW = E_CVRP_TW(path)


'''
Constructive heuristic
'''
RCL_alpha: float = 0.3              # RCL alpha

constructive: Constructive = Constructive(RCL_alpha)


'''
Genetic algorithm
'''
Population_size: int = 3000
Elite_size: int = int(Population_size * 0.05)
crossover_rate: float = 0.6
mutation_rate: float = 0.5

genetic: Genetic = Genetic(Population_size, Elite_size, crossover_rate, mutation_rate)


'''
Repair operators
'''
repair_op: Reparator = Reparator(RCL_alpha)


'''
Feasibility operators
'''
feas_op: Feasibility = Feasibility()


'''
EXPERIMENTATION
Variable convention:
- Population: List of list (individual) of lists (routes) of strings (nodes)
- Distances: List of distances (total per individual)
- Times: List of times (total per individual)
- Details: List of tuples (individual), where the tuple (distances, times) are discriminated per route
- best_individual: list with (individual, distance, time, details)
'''
lab: Experiment = Experiment()
Operators: list = ['simple_crossover', '2opt', 'simple_insertion', 'smart_crossover' , 'Hybrid']
colors: list = ['blue', 'red', 'black', 'purple', 'green', 'orange']
III: list = []

# Hardest = ['c204_21.txt', 'r201_21.txt', 'rc207_21.txt', 'rc208_21.txt']


Results: list = []

for instance in ['c102_21.txt']:
    '''
    Reseting experimentation
    '''
    env.load_data(instance)
    env.generate_parameters()
    
    for operator in [1]:
        
        repair_op.reset(env)

        '''
        Population generation
        '''
        Population, Distances, Times, Details, incumbent, best_individual = genetic.generate_population(env, constructive)
        
        Incumbents: list[float] = [incumbent]
        T_Times: list[float] = [round(time() - start,2)]
        initial_best: list = best_individual
        

        '''
        Evolution
        '''
        # Incumbents, T_Times, Results, incumbent, best_individual = \
        #     lab.evolution(env, genetic, repair_op, Population, Distances, Incumbents, T_Times, Results, best_individual, start, max_time)


    # with open(path + f'Results/{instance}', 'a') as f:
    #     f.write(str(Results))


    # with open(f'/Users/juanbeta/My Drive/2022-2/Metaheurísticas/Tareas/Tarea 4/CG-VRP-TW/Source/Results/R_res_{instance}', 'w') as f:
    #     f.write(str(best_initial) + '\n')
    #     f.write(str(incumbent) + '\n')
    #     f.write(str(best_individual))


    # lab.save_performance(Results, instance, path + f'Results/{instance[:-4]}.png')

print('\n')
print('############## Testing done ################')
print(f'total time: {round(time() - start,2)}')
print(f'incumbent: {round(incumbent,2)}')
print(f'best solution: {initial_best[0]}')
print('\n')
