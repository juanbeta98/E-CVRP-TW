#%%
from E_CVRP_TW import E_CVRP_TW,Constructive,Genetic,Experiment
from time import process_time

experiment_path = '/Users/juanbeta/My Drive/Research/Energy/Experimentation/'

configs = { 'Dar-penalization':'cuadratic',
            'Dar-conservation proportion':0.4,
            'Dar-length restriction':False,
            
            'eval-penalization':'cubic',
            # 'eval-criterion':['Hybrid','phi rate','visited costumers']},
            'eval-criterion':'random',

            'gen-population size':3000,
            'gen-elite size':0.1,
            'gen-crossover rate':0.3,
            'gen-mutation rate':0.3,
                }

# darwinian_configuration = {'penalization':configs['Dar-penalization'],
#                                    'conservation proportion':configs['Dar-conservation proportion'],
#                                    'length restriction':configs['Dar-length restriction']}
# eval_insert_configuration = {'penalization':configs['eval-penalization'],
#                             'criterion':configs['eval-criterion']}

''' Environment '''
env:E_CVRP_TW = E_CVRP_TW()


# ''' Constructive heuristic '''
# constructive:Constructive = Constructive()


# ''' Genetic algorithm '''
# Population_size:int = configs['gen-population size']
# Elite_size:int = int(Population_size*configs['gen-elite size'])

# crossover_rate:float = configs['gen-crossover rate']
# mutation_rate:float = configs['gen-mutation rate']

# genetic: Genetic = Genetic(Population_size,Elite_size,crossover_rate,mutation_rate,
#                            darwinian_configuration,eval_insert_configuration)

# rd_seed = 11
# instance = env.instances[-1]

lab:Experiment = Experiment(experiment_path)

print(' t \t Mean \tMedian \t Std \t Min \t Max \t low \t high')

# lab.HGA(env,constructive,genetic,instance,300,process_time(),True,rd_seed,False)
for instance in env.instances:
    gaps = lab.experiment(instance,configs,verbose=True,save_results=True)





# %%
