#%%
from E_CVRP_TW import E_CVRP_TW,Constructive,Genetic,Experiment
from time import process_time

experiment_path = '/Users/juanbeta/My Drive/Research/Energy/Experimentation/'

configs = {  'init-alpha':0.2,
                  
            'Dar-penalization':'cuadratic',
            'Dar-conservation proportion':0.4,
            'Dar-length restriction':True,
            
            'eval-penalization':'regular',
            # 'eval-criterion':['Hybrid','phi rate','visited costumers']},
            'eval-criterion':'random',

            'gen-population size':500,
            'gen-elite size':0.05,
            'gen-crossover rate':0.3,
            'gen-mutation rate':0.3,
                }

''' Environment '''
env:E_CVRP_TW = E_CVRP_TW()


''' Constructive heuristic '''
constructive:Constructive = Constructive()


''' Genetic algorithm '''
Population_size:int = configs['gen-population size']
Elite_size:int = int(Population_size*configs['gen-elite size'])

crossover_rate:float = configs['gen-crossover rate']
mutation_rate:float = configs['gen-mutation rate']

genetic: Genetic = Genetic(Population_size,Elite_size,crossover_rate,mutation_rate)

rd_seed = 0

lab:Experiment = Experiment(experiment_path,True,True)
lab.HGA(env,constructive,genetic,env.instances[0],30,process_time(),True,0)


# %%
