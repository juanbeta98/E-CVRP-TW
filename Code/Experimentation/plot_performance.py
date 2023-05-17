import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys

def retrieve_const_performance(instance, path):
    file = open(path + f'results_{instance}', 'rb')
    data = pickle.load(file)
    file.close()
    return data


def plot_const_performance(data, instance, testing = 'Constructive', objective = 'min_EV'):
    colors = ['red', 'orange', 'brown', 'green', 'purple' , 'blue' ,'black', 'pink', ]

    if objective == False:
        i = 0
        legend_elements = []
        for key, value in data.items():
            plt.plot(value['inc times'], value['incumbents'], color = colors[i])
            legend_elements.append(Line2D([0], [0], marker='_', color=colors[i], label=str(key), lw=0,
                            markerfacecolor=colors[i], markersize=8))
            i+=1

        plt.title(f"{testing}'s performance {instance}")
        plt.xlabel('Time (s)')
        plt.ylabel('Incumbent')

        ax = plt.gca()
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        plt.show()
    else:
        i = 0
        legend_elements = []
        for key, value in data.items():
            plt.plot(value[objective]['inc times'], value[objective]['incumbents'], color = colors[i])
            legend_elements.append(Line2D([0], [0], marker='_', color=colors[i], label=str(key), lw=0,
                            markerfacecolor=colors[i], markersize=8))
            i+=1

        plt.title(f"{testing}'s performance {instance}")
        plt.xlabel('Time (s)')
        plt.ylabel('Incumbent')

        ax = plt.gca()
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        plt.show()


def retrieve_op_performance(path):
    file = open(path, 'rb')
    data = pickle.load(file)
    file.close()
    return data

def plot_op_performance(data, instance, testing = 'Operators', objective = 'min_EV'):
    colors = ['red', 'orange', 'brown', 'green', 'purple' , 'blue' ,'black', 'pink', 'tab:cyan','tab:olive','tab:gray',
              'lime','lavender','firebrick','palevioletred','magenta','paleturquoise','gold','maroon']
    if objective == 'min_EV':   list_idx = 2
    else:   list_idx = 1

    i = 0
    legend_elements = []
    for key, value in data.items():
        plt.plot(value[list_idx]['inc times'], value[list_idx]['incumbents'], color = colors[i])
        legend_elements.append(Line2D([0], [0], marker='_', color=colors[i], label=f'{key}', lw=0,
                        markerfacecolor=colors[i], markersize=8))
        i+=1

    plt.title(f"{testing}'s performance {instance}")
    plt.xlabel('Time (s)')
    plt.ylabel('Incumbent')

    ax = plt.gca()
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.show()