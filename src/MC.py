# General setup
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker    

import pandas as pd
import numpy as np
import math

from scipy.optimize import curve_fit
from scipy.interpolate import splrep,splev
from scipy import interpolate


# rc Parameters
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['figure.figsize'] = [11, 7]
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16) #Axis labels
plt.rc('xtick', labelsize=14) #Xticks numbers
plt.rc('ytick', labelsize=14) #Yticks numbers
plt.rc('figure', titlesize=16) #Figure title
plt.rc('axes',titlesize=14) #Axes title
plt.rc('legend', fontsize=14) #Legend
plt.rcParams['savefig.bbox'] = 'tight'
ticker.rcParams['axes.formatter.limits']=[-2, 5]



# Leer
def Read(filename):
    df = pd.read_csv(filename, sep='\s+', skiprows=1, float_precision='high', header=None)
    df.rename(columns={1: 'T', 2: 'B', 3: 'M', 9: 'Chi'}, inplace=True)
    
    return df



# Plotear
def Labels(ax, X=' ', Y=' ', figura=''):
    # Grid and ticks
    ax.grid(b=True, axis='both', which='major', color='C7', linestyle='--')
    ax.axes.tick_params(axis='both', which='major', length=3.5, width=2)
    ax.axes.tick_params(axis='both', which='minor', length=2.5, width=1)
    ax.minorticks_on()

    # Labels
    ax.set_xlabel(X)
    ax.set_ylabel(Y)
    ax.legend()

    # Titles
    plt.suptitle('Figure '+figura)
    plt.subplots_adjust(top=0.93)