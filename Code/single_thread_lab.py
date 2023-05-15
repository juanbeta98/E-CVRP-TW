from E_CVRP_TW import E_CVRP_TW, Experiment
# from numpy.random import seed; seed(0)
import os
import itertools

path = f'{os.getcwd()}/'#/Code/'
if path[7:15] == 'juanbeta': computer = 'mac'
else: computer = 'pc'

Operators = ['evaluated insertion']

Configurations = {'Darwinian phi rate':{'penalization':['regular','cuadratic','cubic'],
                                        'conservation proportion':[0.2, 0.4, 0.7],
                                        'length restriction':[True, False],
                                        },
                  
                  'evaluated insertion':{'penalization':['regular','cuadratic','cubic'],
                                         'criterion':['Hybrid', 'phi rate', 'visited costumers']}
                }

# D_keys = list(Configurations['Darwinian phi rate'].keys())
# e_keys = list(Configurations['evaluated insertion'].keys())

# D_combinations = list(itertools.product(*[Configurations['Darwinian phi rate'][key] for key in D_keys]))
# e_combinations = list(itertools.product(*[Configurations['evaluated insertion'][key] for key in e_keys]))

# Grid = [{'Darwinian phi rate': {D_keys[i]: D_combination[i] for i in range(len(D_keys))},'evaluated insertion': {e_keys[i]: e_combination[i] for i in range(len(e_keys))} } for D_combination in D_combinations for e_combination in e_combinations]

# keys = list(Configurations['Darwinian phi rate'].keys())
# combinations = list(itertools.product(*[Configurations['Darwinian phi rate'][key] for key in keys]))
# Grid = [{'Darwinian phi rate': {keys[i]: combination[i] for i in range(len(keys))}} for combination in combinations]


keys = list(Configurations['evaluated insertion'].keys())
combinations = list(itertools.product(*[Configurations['evaluated insertion'][key] for key in keys]))
Grid = [{'evaluated insertion': {keys[i]: combination[i] for i in range(len(keys))}} for combination in combinations]

verbose = True
save_results = False

if __name__ == '__main__':
    env = E_CVRP_TW(path)

    for Configs in Grid:
        Configs = {'evaluated insertion':{'penalization':'regular', 'criterion':'Hybrid'}}
        lab:Experiment = Experiment(path, Operators, Configs, verbose, save_results)

        # test_batch = env.generate_test_batch_per_size(['s','m'],1)
        
        # test_batch = env.generate_test_batch_per_size('l',4)
        # test_batch = test_batch[0]

        for num, instance in enumerate(test_batch):
            progress_percentage = round(round((num+1)/len(test_batch),4)*100,2)
            lab.experimentation(instance, progress_percentage)


