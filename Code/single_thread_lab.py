from E_CVRP_TW import E_CVRP_TW, Experiment
from multiprocess import pool
import os

path = f'{os.getcwd()}/'#Code/'
verbose = True
save_results = False

if __name__ == '__main__':
    env = E_CVRP_TW(path)
    lab:Experiment = Experiment(path, verbose, False, save_results)

    # test_bed = [env.sizes['s'][0],env.sizes['m'][0],env.sizes['l'][0]]
    test_bed = ['r202C15.txt']

    for num, instance in enumerate(test_bed):
        progress_percentage = round(round((num+1)/len(test_bed),4)*100,2)
        lab.experimentation(instance, progress_percentage)

