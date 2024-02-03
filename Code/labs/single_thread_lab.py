from E_CVRP_TW import E_CVRP_TW, Experiment
# from numpy.random import seed; seed(0)
import os
from itertools import product

experiments_path = '/Users/juanbeta/My Drive/Research/Energy/Experimentation/'

Configurations = {  'init-alpha':[0.2,0.4,0.6],
                  
                    'Dar-penalization':['cuadratic','cubic'],
                    'Dar-conservation proportion':[0.4],
                    'Dar-length restriction':[True, False],
                  
                    'eval-penalization':['regular','cuadratic','cubic'],
                    # 'eval-criterion':['Hybrid','phi rate','visited costumers']},
                    'eval-criterion':['random'],

                    'gen-population size':[500,1000,2000,3000],
                    'gen-elite size':[0.05,0.15,0.25],
                    'gen-crossover rate':[0.3,0.6],
                    'gen-mutation rate':[0.3,0.6]
                }

# Generate all combinations
combinations = product(*(Configurations[key] for key in Configurations))

# Create a list of dictionaries with the combinations
Grid = [dict(zip(Configurations.keys(), combo)) for combo in combinations]

verbose = False
save_results = True


if __name__ == '__main__':
    env = E_CVRP_TW()

    for num,Configs in enumerate(Grid):

        ### Print progress of experimentation
        progress_percentage = round(round((num+1)/len(Grid),4)*100,2)
        print(f'\n-------- Experiment {num} / {progress_percentage}% --------')


        ### Save README file with experimentation details
        with open(experiments_path+f'/Third phase/Exp {num}/readme.txt', 'w') as f:
            readme = f'Experiment {num}'
            readme += f'\nInit: \t{Configs["Initialization"]["alpha"]}'
            readme += f'\nDarwinian phi rate: \t{Configs["Darwinian phi rate"]["penalization"]} - {Configs["Darwinian phi rate"]["length restriction"]}'
            readme += f'\nevaluated insertion: \t{Configs["evaluated insertion"]["penalization"]} - {Configs["evaluated insertion"]["criterion"]}'
            readme += f'\ngenetic configuration: \t{Configs["genetic parameters"]["population size"]} - {Configs["genetic parameters"]["crossover rate"]} - {Configs["genetic parameters"]["mutation rate"]}'
            f.write(readme)
                
    


        lab:Experiment = Experiment(experiments_path,Configs,verbose,save_results,num)

        for i,instance in enumerate(env.instances):
            Results = lab.experimentation(instance)



