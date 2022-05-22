import os
#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


from keras.layers import LSTM, Dense
from keras.models import Sequential, load_model
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np




def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


def split_series(series, n_past, n_future):
	#
	# n_past ==> no of past observations
	#
	# n_future ==> no of future observations
	#
	X, y = list(), list()
	for window_start in range(len(series)):
		past_end = window_start + n_past
		future_end = past_end + n_future
		if future_end > len(series):
			break
		# slicing the past and future parts of the window
		past, future = series[window_start:past_end, :], series[past_end:future_end, :]
		X.append(past)
		y.append(future)
	return np.array(X), np.array(y)


data = read_csv('dataset.csv')

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(data.values)
all_x, all_y = split_series(scaled, 4, 1)



n_train = 90
train_x = all_x[:n_train, :, :]
test_x = all_x[n_train:, :, :]
# trying to predict the open price
train_y = all_y[:n_train, :, 8]
# trying to predict the open price
test_y = all_y[n_train:, :, 8]



if os.path.isdir('model'):
	model = load_model('model')
else:
	# design network
	model = Sequential()
	model.add(LSTM(50, input_shape=(train_x.shape[1], train_x.shape[2])))
	model.add(Dense(1))
	model.compile(loss='mae', optimizer='adam')
	# fit network
	history = model.fit(
		train_x,
		train_y,
		epochs=220,
		batch_size=4,
		validation_data=(test_x, test_y),
		verbose=2,
		shuffle=False
	)
	# plot history
	plt.plot(history.history['loss'], label='train')
	plt.plot(history.history['val_loss'], label='test')
	plt.legend()
	plt.show()
	# save model
	model.save('model')




y_pred = model.predict(test_x)
real_y = test_y
plt.plot(y_pred, label='prediction')
plt.plot(real_y, label='reality')
plt.legend()
plt.show()


average_error = 0
for i in range(len(y_pred)):
	average_error += abs(1 - real_y[i]/y_pred[i])
average_error /= len(y_pred)

print(f'average error: {average_error * 100}%')
