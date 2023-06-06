from E_CVRP_TW import E_CVRP_TW, Experiment
from numpy.random import seed; seed(0)
from multiprocess import pool
import os
import itertools
from time import process_time

path = f'{os.getcwd()}/'
saving_path = os.path.abspath('/')
saving_path += (f'Documents/Experimentation 1.0/')

full_capacity = True

Operators = ['Darwinian phi rate', 'evaluated insertion']

Configurations = {'Darwinian phi rate':{'penalization':['regular', 'cuadratic','cubic'],
                                        'conservation proportion':[0.25, 0.5, 0.7],
                                        'length restriction':[True, False],
                                        },
                  
                  'evaluated insertion':{'penalization':['regular','cuadratic','cubic'],
                                         'criterion':['visited costumers', 'phi rate', 'Hybrid', 'random']},

                  'genetic parameters':{'population size':[850,2000,3500],
                                        'crossover rate':[0.25, 0.5, 0.7],
                                        'mutation rate':[0.2, 0.4, 0.7]}
                }

D_keys = list(Configurations['Darwinian phi rate'].keys()); D_combinations = list(itertools.product(*[Configurations['Darwinian phi rate'][key] for key in D_keys]))
e_keys = list(Configurations['evaluated insertion'].keys()); e_combinations = list(itertools.product(*[Configurations['evaluated insertion'][key] for key in e_keys]))
g_keys = list(Configurations['genetic parameters'].keys()); g_combinations = list(itertools.product(*[Configurations['genetic parameters'][key] for key in g_keys]))

Grid = [{'Darwinian phi rate': {D_keys[i]: D_combination[i] for i in range(len(D_keys))}, 
         'evaluated insertion': {e_keys[i]: e_combination[i] for i in range(len(e_keys))}, 
         'genetic parameters': {g_keys[i]: g_combination[i] for i in range(len(g_keys))}} for D_combination in D_combinations for e_combination in e_combinations for g_combination in g_combinations]


env = E_CVRP_TW(path)
test_batch = env.instances

if __name__ == '__main__':
    for num, Configs in enumerate(Grid): 
        os.mkdir(path + f'Exp {num}')
        with open(path + f'Exp {num}/readme.txt', 'w') as f:
            readme = f'Experiment {num}'
            readme += f'\nDarwinian phi rate: \t{Configs["Darwinian phi rate"]["penalization"]} - {Configs["Darwinian phi rate"]["length restriction"]}'
            readme += f'\nevaluated insertion: \t{Configs["evaluated insertion"]["penalization"]} - {Configs["evaluated insertion"]["criterion"]}'
            readme += f'\ngenetic configuration: \t{Configs["genetic parameters"]["population size"]} - {Configs["genetic parameters"]["crossover rate"]} - {Configs["genetic parameters"]["mutation rate"]}'
            f.write(readme)
                

        progress_percentage = round(round((num+1)/len(Grid),4)*100,2)
        iter_start = process_time()
        print(f'\n-------- Experiment {num} / {progress_percentage}% --------')

        lab:Experiment = Experiment(path, Operators, Configs, False, True, num, saving_path)

        if not full_capacity:   p = pool.Pool(processes = 8)
        else: p = pool.Pool()

        Results = p.map(lab.experimentation, test_batch)
        print(f'Average gap: {round(sum(Results)/len(Results),2)}')
        print(f'Running time: {round(process_time() - iter_start)}')

        p.terminate()
