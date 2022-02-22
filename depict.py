import numpy as np 
from matplotlib import pyplot as plt

""" 
This file helps plot different graphs
"""

def BasicPlot(y, file_name='noname.png',x_text = "", y_text="", title_text="",colour='blue', scale=1, x_shift=0):
    plt.figure();
    x=np.array(range(len(y)))+1+x_shift;  x*=scale;
    plt.plot(x,y,color=colour,linewidth=0.7);
    plt.xlabel(x_text);
    plt.ylabel(y_text);
    plt.title(title_text);
    plt.savefig(file_name, bbox_inches='tight');


def MultiplePlot(y_list, file_name='noname.png',x_text = "", y_text="", title_text="", line_label=None, colour=None, scale=1, x_shift=0):
    
    plt.figure();
    plt.xlabel(x_text);
    plt.ylabel(y_text);
    x=np.array(range(len(y_list[0])))+1+x_shift;  x*=scale;
    for i in range(len(y_list)):
        plt.plot(x,y_list[i], linewidth=0.5, color=colour[i], label=line_label[i]);
        
    plt.legend();    
    plt.title(title_text);
    plt.savefig(file_name, bbox_inches='tight');