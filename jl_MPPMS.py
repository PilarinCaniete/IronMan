# General setup
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker    

import pandas as pd
import numpy as np
import math
import os

from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

from ipywidgets import interactive, FloatSlider


# rc Parameters
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams['figure.figsize'] = [11, 5]
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
    return pd.read_csv(filename, index_col=False, skiprows=7, float_precision='high')

    

# Parámetros
def temperature(filename):
    df = Read(filename)

    return df['Temp_mean'].mean()



# Factor demagnetizante
def D_prima(a,b,c):
    return ( (b**2-c**2)/(2*b*c) * np.log((np.sqrt(a**2+b**2+c**2)-a)/(np.sqrt(a**2+b**2+c**2)+a))  +
(a**2-c**2)/(2*a*c) * np.log((np.sqrt(a**2+b**2+c**2)-b)/(np.sqrt(a**2+b**2+c**2)+b))  +
b/(2*c) * np.log((np.sqrt(a**2+b**2)+a)/(np.sqrt(a**2+b**2)-a))  +  a/(2*c) * np.log((np.sqrt(a**2+b**2)+b)/(np.sqrt(a**2+b**2)-b))  +
c/(2*a) * np.log((np.sqrt(b**2+c**2)-b)/(np.sqrt(b**2+c**2)+b))  +  c/(2*b) * np.log((np.sqrt(a**2+c**2)-a)/(np.sqrt(a**2+c**2)+a))  +
2 * np.arctan((a*b)/(c*np.sqrt(a**2+b**2+c**2)))  +
(a**3+b**3-2*c**3)/(3*a*b*c)  +  (a**2+b**2-2*c**2)/(3*a*b*c) * np.sqrt(a**2+b**2+c**2)  +
c/(a*b) * (np.sqrt(a**2+c**2)+np.sqrt(b**2+c**2))  -
((a**2+b**2)**(3/2)+(b**2+c**2)**(3/2)+(c**2+a**2)**(3/2))/(3*a*b*c) ) / np.pi


def D(a,b,c):
    return D_prima(a,b,c) * 16/(10e-10)**3 * np.pi*4e-7 * 10*9.274e-24



# Unidades
def UnitsDy(df, dim, mass):
    df['B'] = df['Field_Oe'] / 10000
    df['M'] = df['Moment_long'] / (mass/1000) * 532.73/2 / (6.022E23) / (0.9274E-20) / 10
    df['B_real'] = df['B'] - D(*dim)*df['M']


def UnitsHo(df, dim, mass):
    df['B'] = df['Field_Oe'] / 10000
    df['M'] = df['Moment_long'] / (mass/1000) * 537.59/2 / (6.022E23) / (0.9274E-20) / 10
    df['B_real'] = df['B'] - D(*dim)*df['M']



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



# Análisis
## Derivada:
from scipy.interpolate import splrep,splev

def Derivada(df, X, Y):
    # Process
    df_fix = df[df.duplicated('Field_Oe', keep='first')==False]
    df_fix = df_fix.sort_values('Field_Oe', axis=0, ascending=True)

    # Interpolation and derivative
    tck = splrep(df_fix[X], df_fix[Y], s=0)
    x = np.linspace(df_fix[X].min(), df_fix[X].max(), len(df_fix.index))
    y = splev(x, tck, der=1)

    return pd.DataFrame({'H':x, 'Chi':y})
