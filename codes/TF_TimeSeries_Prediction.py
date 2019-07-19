
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
import tensorflow as tf 
from tensorflow import keras

# plot function
def plot_data(time, series, format = "-",
                start = 0, end = None):
    plt.figure(figsize = (10,6))
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()

def plot_multiplot(time1, series1, time2, series2):
    plt.figure(figsize = (10,6))
    plt.subplot(211)
    plt.plot(time1, series1)
    plt.xlim(left=min(time1)-10, right=max(time2)+10)
    plt.ylim(bottom= min(series1), top=max(series2))
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.subplot(212)
    plt.plot(time2, series2)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.xlim(left=min(time1)-10, right=max(time2)+10)
    plt.ylim(bottom=min(series1), top=max(series2))
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
    #pattern is repeated according to period
    return amplitude * season_pattern(season_time)

def noise(time, noise_level = 1, seed = None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

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

def create_dataset(time, slope, amplitude, period, noise_level,\
    					baseline, seed, split_no):
	
	series = baseline + trend(time, slope)\
            + seasonality(time, period = period, amplitude = amplitude)\
            + noise(time, noise_level = noise_level, seed = seed)
	
	train_time = time[:split_no]
	train_series = series[:split_no]

	validation_time = time[split_no:]
	validation_series = series[split_no:]

	return series, train_time, train_series, validation_time, validation_series

def windowed_dataset(series, window_size, batch_size, shuffle_buffer_size):
  # tf.data.Dataset.from_tensor_slices():
  #Creates a Dataset whose elements are slices of the given tensors.
	dataset = tf.data.Dataset.from_tensor_slices(series)
	dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
	dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
	dataset = dataset.shuffle(shuffle_buffer_size).map\
				(lambda window: (window[:-1], window[-1]))
	dataset = dataset.batch(batch_size).prefetch(1)
	return dataset

########################################################################
# The main() function
def main():
    
    print(tf.__version__)

    
    time = np.arange(4 * 365 + 1, dtype = "float32")
    #arange: Return evenly spaced values within a given interval.
    #default start = 0, 4 * 365 + 1: stop number.
    slope = 0.05
    amplitude = 40
    period = 365
    noise_level = 5
    baseline = 10
    seed = 42
    split_no = round(len(time) * 0.8) # 80% training data
    window_size = 20
    batch_size = 32
    shuffle_buffer_size = 1000
    epochs=100

	### ----- Create a time series training - validation dataset -----
    series, train_time, train_series, validation_time, validation_series =\
    		create_dataset(time, slope, amplitude, period, noise_level,\
    					baseline, seed, split_no)
    # plot_data(train_time, train_series)
    # plot_data(validation_time, validation_series)
    # plot_multiplot(train_time, train_series, validation_time, validation_series)

    dataset = windowed_dataset(train_series, window_size, batch_size,\
 							shuffle_buffer_size)
    # print(dataset)
    ### ----- Create and train the model -----
    l0 = tf.keras.layers.Dense(1, input_shape=[window_size])
    model = tf.keras.models.Sequential([l0])
    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))
    model.fit(dataset,epochs=epochs,verbose=1)
    # print("Layer weights {}".format(l0.get_weights()))

    ### ----- prediction -----
    forecast = []
    for time in range(len(series) - window_size):
    	forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

    forecast = forecast[split_no-window_size:]
    results = np.array(forecast)[:, 0, 0]

    ### ----- plot the result -----
    fig, ax = plt.subplots()
    ax.plot(validation_time, validation_series, color='k', label="Real data")
    ax.plot(validation_time, results, color ='g', label = "Prediction")
    plt.legend()
    plt.show()

#######################################################################
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()

