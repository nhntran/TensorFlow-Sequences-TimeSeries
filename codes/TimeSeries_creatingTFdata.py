
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



########################################################################
# The main() function
def main():
    
    # Create a simple tf dataset: 10 elements from 0 to 9
    # dataset = tf.data.Dataset.range(10)
    # for val in dataset:
    #     print(val.numpy())

    #Create a dataset with shift window: 
    # dataset = tf.data.Dataset.range(10)
    # dataset = dataset.window(5, shift=1)
    # for window_dataset in dataset:
    #     for val in window_dataset:
    #         print(val.numpy(), end=";")
    #     print()
    # Result:
    # 0;1;2;3;4;
    # 1;2;3;4;5;
    # 2;3;4;5;6;
    # 3;4;5;6;7;
    # 4;5;6;7;8;
    # 5;6;7;8;9;
    # 6;7;8;9;
    # 7;8;9;
    # 8;9;
    # 9;

    #Create a dataset with shift window 6x5: 
    # dataset = tf.data.Dataset.range(10)
    # dataset = dataset.window(5, shift=1, drop_remainder = True)
    # for window_dataset in dataset:
    #     for val in window_dataset:
    #         print(val.numpy(), end=";")
    #     print()
    # Result:
    # 0;1;2;3;4;
    # 1;2;3;4;5;
    # 2;3;4;5;6;
    # 3;4;5;6;7;
    # 4;5;6;7;8;
    # 5;6;7;8;9;   

    #Create a dataset with shift window 6 arrays of 5: 
    # dataset = tf.data.Dataset.range(10)
    # dataset = dataset.window(5, shift=1, drop_remainder = True)
    # dataset = dataset.flat_map(lambda window:window.batch(5))
    # for window in dataset:
    #     print(window.numpy())
            
    # Result:
    # [0 1 2 3 4]
    # [1 2 3 4 5]
    # [2 3 4 5 6]
    # [3 4 5 6 7]
    # [4 5 6 7 8]
    # [5 6 7 8 9]

    #Create a dataset with x and y in dataset: 
    # dataset = tf.data.Dataset.range(10)
    # dataset = dataset.window(5, shift=1, drop_remainder = True)
    # dataset = dataset.flat_map(lambda window:window.batch(5))
    # dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
    # dataset = dataset.shuffle(buffer_size=10) #randomize the row only
    # for x,y in dataset:
    #     print(x.numpy(), y.numpy())
    # Result:
    # [4 5 6 7] [8]
    # [2 3 4 5] [6]
    # [5 6 7 8] [9]
    # [0 1 2 3] [4]
    # [1 2 3 4] [5]
    # [3 4 5 6] [7]

    #Create a dataset with x and y in dataset: 
    # dataset = tf.data.Dataset.range(10)
    # dataset = dataset.window(5, shift=1, drop_remainder = True)
    # dataset = dataset.flat_map(lambda window:window.batch(5))
    # dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
    # dataset = dataset.shuffle(buffer_size=10) #randomize the row only
    # dataset = dataset.batch(2).prefetch(1)
    # # array of array of 2
    # for x,y in dataset:
    #     print(x.numpy(), y.numpy())
    # Result:
    # [[2 3 4 5][1 2 3 4]] [[6][5]]
    # [[5 6 7 8][0 1 2 3]] [[9][4]]
    # [[4 5 6 7][3 4 5 6]] [[8][7]]

    # Create a dataset with shift window:
    # Dataset: x: row x col, y: row x 1
    range_no = 10 #row + col for x = 7 x 3
    window_size = 4 #col +1
    dataset = tf.data.Dataset.range(range_no)
    dataset = dataset.window(window_size, shift=1, drop_remainder = True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.shuffle(buffer_size=range_no)
    for x,y in dataset:
        print(x.numpy(), y.numpy())
    # Result:
    # [0 1 2] [3]
    # [1 2 3] [4]
    # [2 3 4] [5]
    # [3 4 5] [6]
    # [4 5 6] [7]
    # [5 6 7] [8]
    # [6 7 8] [9]
#######################################################################
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()

