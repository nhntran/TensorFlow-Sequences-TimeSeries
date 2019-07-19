Deep Learning with TensorFlow - Sequences and Time Series - Tutorials
================
Codes courtesy from TensorFlow in Practice Specialization by deeplearning.ai on Coursera, modified by Tran Nguyen.

Quick notes from the courses + codes to run in Mac terminal. If you want to learn more about TensorFlow, check out the great courses in the "TensorFlow in Practice Specialization" by deeplearning.ai on Coursera.

The codes work well with TensorFlow 2.0

Ref: <https://github.com/lmoroney/dlaicourse/tree/master/TensorFlow%20In%20Practice>

``` bash
pip install tensorflow==2.0.0-alpha0
pip install tf-nightly-2.0-preview #for generate TF dataset 
```

#### 1. Basic about Time Series

-   Codes: TimeSeries\_basic.py
-   What you will learn: (i) Create a time series and different transformation on time series. (ii) Plot using matplotlib.
-   Reference about matplotlib: <https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html>

#### 2. Create TF dataset

-   Codes: TimeSeries\_creatingTFdata.py
-   What you will learn: (i) Create a simple TF dataset using a range of number.

``` r
# Create a dataset that have x: 7 rows, 3 columns; corresponding to y: 7 rows, 1 column
# Define range_no = row + col in x, window_size = col in x + 1
# For example: generate a TF dataset of x: 7 x 3 and y: 7 x 1
range_no = 7 + 3 #row + col for x = 7 x 3
window_size = 3 + 1 #col +1
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
```

#### 3. Create the data and TF prediction

-   Codes: TF\_TimeSeries\_Prediction.py
-   What you will learn: (i) prediction the time series data using the TF model. (ii) Plot the result and comparing the real - prediction data using overlaying plot (matplotlib.pyplot)

``` r
    l0 = tf.keras.layers.Dense(1, input_shape=[window_size])
    model = tf.keras.models.Sequential([l0])
    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))
    model.fit(dataset,epochs=epochs,verbose=1)
```

-   The plot result:

<img src="/Users/trannguyen/TranData/WORK/BioinformaticsSpecialization_Tran_2019/MachineLearning/TensorFlow_Sequences_TimeSeries/img/TimeSeries_Prediction.png" width="1059" style="display: block; margin: auto;" />
