import pickle
import matplotlib.pyplot as plt
import sys

def retrieve_const_performance(instance, path):

    file = open(path + f'results_{instance}', 'rb')
    data = pickle.load(file)
    file.close()

    return data


def plot_performance(data, instance, testing = 'Constructive'):
    colors = ['red', 'orange', 'brown', 'green', 'purple' , 'blue' ,'black', 'pink', ]
    i = 0
    for key, value in data.items():
        plt.plot(value['inc times'], value['incumbents'], color = colors[i]);i+=1
    plt.title(f"{testing}'s performance {instance}")
    plt.xlabel('Time (s)')
    plt.ylabel('Incumbent')

    plt.show()
