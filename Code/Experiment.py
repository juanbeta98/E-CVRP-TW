from numpy.random import seed, choice, random
from E_CVRP_TW import  E_CVRP_TW, Constructive, Genetic, Feasibility, Reparator, Experiment
from time import time

'''
General parameters
'''
rd_seed: int = 0
seed(rd_seed)
path = '/Users/juanbeta/My Drive/Research/Energy/CG-VRP-TW/' ##### CHANGE IF NECESSARY!!!


'''
Environment
'''
env: E_CVRP_TW = E_CVRP_TW(path)


'''
Constructive heuristic
'''
RCL_alpha: float = 0.35             # RCL alpha
End_slack: int = 20                 # Slack to send veicles to depot
constructive: Constructive = Constructive(RCL_alpha, End_slack)


'''
Genetic algorithm
'''
Population_size: int = 2000  
Elite_size: int = int(Population_size * 0.05)

genetic: Genetic = Genetic(Population_size, Elite_size)

max_time: int = 3600
crossover_rate: float = 0.6
mutation_rate: float = 0.5


'''
Repair operators
'''
repair_op: Reparator = Reparator()


'''
Feasibility operators
'''
feas_op: Feasibility = Feasibility()



'''
EXPERIMENT
'''
lab = Experiment()
Operators = ['simple_crossover', '2opt', 'simple_insertion', 'smart_crossover' , 'Hybrid']
colors = ['blue', 'red', 'black', 'purple', 'green', 'orange']
III = []

# Hardest = ['c204_21.txt', 'r201_21.txt', 'rc207_21.txt', 'rc208_21.txt']
env = CG_VRP_TW()
Instances = ['c101C10.txt',
 'c101C5.txt',
 'c101_21.txt',
 'c102_21.txt',
 'c103C15.txt',
 'c103C5.txt',
 'c103_21.txt',
 'c104C10.txt',
 'c105_21.txt',
 'c106C15.txt',
 'c107_21.txt',
 'c108_21.txt',
 'c109_21.txt',
 'c201_21.txt',
 'c202C10.txt',
 'c202C15.txt',
 'c203_21.txt',
 'c204_21.txt',
 'c205C10.txt',
 'c205_21.txt',
 'c206C5.txt',
 'c206_21.txt',
 'c208C5.txt',
 'c208_21.txt',
 'r101_21.txt',
 'r102C10.txt',
 'r102C15.txt',
 'r102_21.txt',
 'r103C10.txt',
 'r103_21.txt',
 'r104C5.txt',
 'r104_21.txt']


Results = []

for instance in ['c102_21.txt']:
    '''
    Reseting experimentation
    '''
    env.load_data(instance)
    env.generate_parameters()
    
    for operator in Operators:
        
        repair_op.reset(env)

        best_obj = 1e9
        best_ind = []


        '''
        Evolution
        '''
        Population, Distances, Times, best_individual, incumbent, Incumbents, TTimes, start = lab.generate_intial_population(env, constructive, genetic, Population_size, RCL_alpha, End_slack)
        best_initial = incumbent
        Incumbents, TTimes, Results, best_individual, incumbent = \
            lab.evolution(Population, Distances, Times, Incumbents, TTimes, Results, best_individual, incumbent, start, env, genetic, repair_op, Population_size, Elite_size, max_time,  RCL_alpha, End_slack, crossover_rate)


    with open(path + f'Results/{instance}', 'a') as f:
        f.write(str(Results))


    # with open(f'/Users/juanbeta/My Drive/2022-2/Metaheurísticas/Tareas/Tarea 4/CG-VRP-TW/Source/Results/R_res_{instance}', 'w') as f:
    #     f.write(str(best_initial) + '\n')
    #     f.write(str(incumbent) + '\n')
    #     f.write(str(best_individual))


    lab.save_performance(Results, instance, path + f'Results/{instance[:-4]}.png')



