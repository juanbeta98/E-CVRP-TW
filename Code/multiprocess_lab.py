from E_CVRP_TW import E_CVRP_TW, Experiment
from multiprocess import pool
import os
import itertools

path = f'{os.getcwd()}/'
full_capacity = input('Test on full capacity? [Y/n] ')
if path[7:15] == 'juanbeta' and full_capacity: computer = 'mac'
else: computer = 'pc'

Operators = ['Darwinian phi rate']

Configurations = {'Darwinian phi rate':{'penalization':['regular','cuadratic','cubic'],
                                        'conservation proportion':[0.2, 0.4, 0.7],
                                        'length restriction':[True, False],
                                        },
                  
                  'evaluated insertion':{'criterion':['Hybrid', 'phi rate', 'visited costumers']}
                }

# D_keys = list(Configurations['Darwinian phi rate'].keys())
# e_keys = list(Configurations['evaluated insertion'].keys())

# D_combinations = list(itertools.product(*[Configurations['Darwinian phi rate'][key] for key in D_keys]))
# e_combinations = list(itertools.product(*[Configurations['evaluated insertion'][key] for key in e_keys]))

# Grid = [{'Darwinian phi rate': {D_keys[i]: D_combination[i] for i in range(len(D_keys))},'evaluated insertion': {e_keys[i]: e_combination[i] for i in range(len(e_keys))} } for D_combination in D_combinations for e_combination in e_combinations]


keys = list(Configurations['Darwinian phi rate'].keys())
combinations = list(itertools.product(*[Configurations['Darwinian phi rate'][key] for key in keys]))
Grid = [{'Darwinian phi rate': {keys[i]: combination[i] for i in range(len(keys))}} for combination in combinations]


if __name__ == '__main__':
    env = E_CVRP_TW(path)

    for num, Configs in enumerate(Grid):
        testing_config = str()
        for vals in list(Configs[Operators[0]].values()): testing_config += str(vals)+'-'
        testing_config = testing_config[:-1]

        progress_percentage = round(round((num+1)/len(Grid),4)*100,2)

        print(f'-------------- {Operators[0]} / {testing_config} / {progress_percentage}% --------------')

        lab:Experiment = Experiment(path, Operators, Configs, False, True)

        if computer == 'mac':   p = pool.Pool(processes = 8)
        else: p = pool.Pool()

        p.map(lab.experimentation, env.instances)
        p.terminate()