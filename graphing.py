from matplotlib import pyplot
from pandas import read_csv


dataset = read_csv('dataset.csv', header=0, index_col=0)
values = dataset.values

groups = range(13)
i = 1
pyplot.figure()
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group])
	pyplot.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
pyplot.show()