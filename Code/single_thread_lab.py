from E_CVRP_TW import E_CVRP_TW, Experiment
import os

path = f'{os.getcwd()}/'#Code/'
if path[7:15] == 'juanbeta': computer = 'mac'
else: computer = 'pc'

verbose = True
save_results = False

if __name__ == '__main__':
    env = E_CVRP_TW(path)
    lab:Experiment = Experiment(path, verbose, save_results)

    test_bed = env.generate_test_batch(computer)

    for num, instance in enumerate(test_bed):
        progress_percentage = round(round((num+1)/len(test_bed),4)*100,2)
        lab.experimentation(instance, progress_percentage)

