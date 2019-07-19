
 #!/usr/bin/env python3
 # -*- coding: utf-8 -*-

### Learning TensorFlow with Laurence Moroney, Google Brain on Coursera
#/Users/trannguyen/TranData/WORK/BioinformaticsSpecialization_Tran_2019/
#MachineLearning/TensorFlow_Sequences_TimeSeries/codes/

# Reference: 
#https://github.com/lmoroney/dlaicourse/tree/master/TensorFlow%20In%20Practice

# Reference about matplotlib:
#https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html

# Need to use tensorflow 2.0.0
# Check version with 'print(tf.__version__)'
#'pip install tensorflow==2.0.0-beta0' for install
# if it's not 2.00 => include this line of code:
# 'tf.enable_eager_execution()'

import numpy as np
import matplotlib.pyplot as plt

# plot function
def plot_data(time, series):
    plt.figure(figsize = (10,6))
    plt.plot(time, series)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()

def trend(time, slope=0):
    return slope * time 
# createing a pattern for season
# if time<0.4: cos function, else: exp function
def season_pattern(time):
    return np.where(time<0.4,
                np.cos(time*2*np.pi),
                1/np.exp(3*time))

def seasonality(time, period, amplitude = 1, phase = 0):
    season_time = ((time+phase)%period)/period
    return amplitude * season_pattern(season_time)

def noise(time, noise_level = 1):
    return np.random.randn(len(time)) * noise_level

def autocorrelation1(time, amplitude):
    rho1 = 0.5
    rho2 = -0.1
    ar = np.random.rand(len(time) +50)
    ar[:50] = 100
    for step in range(50, len(time)+50):
        ar[step] += rho1 * ar[step -50]
        ar[step] += rho2 * ar[step -33]
    return ar[50:] * amplitude

def autocorrelation2(time, amplitude):
    rho = 0.8
    ar = np.random.rand(len(time) +1)
    for step in range(1, len(time)+1):
        ar[step] += rho * ar[step - 1]
    return ar[1:] * amplitude

########################################################################
# The main() function
def main():
    
    time = np.arange(4 * 365 + 1)
    #arange: Return evenly spaced values within a given interval.
    #default start = 0, 4 * 365 + 1: stop number.

    #1. basic line
    slope = 0.1
    series1 = trend(time, slope)

    #2. Including seasonality
    amplitude = 40
    series2 = seasonality(time, period = 365, amplitude = amplitude)
    
    #plot_data(time, series)
    #plot
    plt.figure(figsize = (10,6))
    #plot1
    plt.subplot(121)
    #the number of rows, the number of columns
    #and the index of the plot
    plt.plot(time, series1)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

    #plot2
    plt.subplot(122)
    plt.plot(time, series2)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()

    #3. Add baseline
    baseline = 10
    series3 = series1 + series2 + baseline

    #4. Add noise
    noise_level = 10
    series4 = series3 + noise(time, noise_level)
    
    #5. Add autocorrelation1
    series5 = series4 + autocorrelation1(time, 10)
    
    #6. Add autocorrelation2
    series6 = series4 + autocorrelation2(time, 10)

    

    
    plt.figure(figsize = (10,6))
    #plot3
    plt.subplot(221)
    plt.plot(time, series3)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
   
    #plot4
    plt.subplot(222)
    plt.plot(time, series4)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    #plot5
    plt.subplot(223)
    plt.plot(time, series5)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    #plot6
    plt.subplot(224)
    plt.plot(time, series6)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

    plt.show()


#######################################################################
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()

