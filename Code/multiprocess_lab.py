from E_CVRP_TW import E_CVRP_TW, Experiment
from numpy.random import seed; seed(0)
from multiprocess import pool
import os
import itertools

path = f'{os.getcwd()}/'

if path[7:15] == 'juanbeta':
    full_capacity = input('Test on full capacity? [Y/n] ')
    if full_capacity: computer = 'mac'
    else: computer = 'pc'
else: computer = 'pc'

Operators = ['evaluated insertion']

Configurations = {'Darwinian phi rate':{'penalization':['cuadratic','cubic'],
                                        'conservation proportion':[0.4],
                                        'length restriction':[True, False],
                                        },
                  
                  'evaluated insertion':{'penalization':['regular','cuadratic','cubic'],
                                         #'criterion':['Hybrid', 'phi rate', 'visited costumers']},
                                         'criterion':['random']},

                  'genetic parameters':{'population size':[1500,3000],
                                        'crossover rate':[0.3, 0.6],
                                        'mutation rate':[0.3, 0.6]}
                }

D_keys = list(Configurations['Darwinian phi rate'].keys()); D_combinations = list(itertools.product(*[Configurations['Darwinian phi rate'][key] for key in D_keys]))
e_keys = list(Configurations['evaluated insertion'].keys()); e_combinations = list(itertools.product(*[Configurations['evaluated insertion'][key] for key in e_keys]))
g_keys = list(Configurations['genetic parameters'].keys()); g_combinations = list(itertools.product(*[Configurations['genetic parameters'][key] for key in g_keys]))

# Grid = [{'Darwinian phi rate': {D_keys[i]: D_combination[i] for i in range(len(D_keys))}, 
#          'evaluated insertion': {e_keys[i]: e_combination[i] for i in range(len(e_keys))}, 
#          'genetic parameters': {g_keys[i]: g_combination[i] for i in range(len(g_keys))}} for D_combination in D_combinations for e_combination in e_combinations for g_combination in g_combinations]

Grid = [{'evaluated insertion': {e_keys[i]: e_combination[i] for i in range(len(e_keys))}} for e_combination in e_combinations]




if __name__ == '__main__':
    env = E_CVRP_TW(path)

    for num, Configs in enumerate(Grid):
        # with open(path + f'Experimentation/Experiment {num}/readme.txt', 'w') as f:
        #     readme = f'Experiment {num}'
        #     readme += f'\nDarwinian phi rate: \t{Configs["Darwinian phi rate"]["penalization"]} - {Configs["Darwinian phi rate"]["length restriction"]}'
        #     readme += f'\nevaluated insertion: \t{Configs["evaluated insertion"]["penalization"]} - {Configs["evaluated insertion"]["criterion"]}'
        #     readme += f'\ngenetic configuration: \t{Configs["genetic parameters"]["population size"]} - {Configs["genetic parameters"]["crossover rate"]} - {Configs["genetic parameters"]["mutation rate"]}'
        #     f.write(readme)

        progress_percentage = round(round((num+1)/len(Grid),4)*100,2)
        print(f'\n-------- Experiment {num} / {progress_percentage}% --------')

        lab:Experiment = Experiment(path, Operators, Configs, False, True, None)

        if computer == 'mac':   p = pool.Pool(processes = 8)
        else: p = pool.Pool()

        p.map(lab.experimentation, env.instances)
        p.terminate()
