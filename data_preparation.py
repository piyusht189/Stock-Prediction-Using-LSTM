import operator
import csv



stats = {}
THRESHOLD = 0.8
with open('hydrated_tweets2.csv', encoding="utf8") as csvfile:
    hydrated_tweets = csv.reader(csvfile, delimiter=',')
    first = True
    for row in hydrated_tweets:
        if first:
            first = False
            continue
        date = row[-1].split(' ')
        if len(date[1]) == 1:
            date[1] = '0' + date[1]
        if len(date[2]) == 1:
            date[2] = '0' + date[2]
        date = int(''.join(date))
        # 1 counting for the tweet itself
        retweet = int(row[-6]) + 1
        neu = float(row[-2]) * retweet
        neg = float(row[-3]) * retweet
        pos = float(row[-4]) * retweet
        number_neu = retweet if neu > THRESHOLD else 0
        number_pos = retweet if pos > THRESHOLD else 0
        number_neg = retweet if neg > THRESHOLD else 0
        stat = (pos, neg, neu, number_pos, number_neg, number_neu, retweet)
        stats[date] = tuple(map(operator.add, stat, stats.get(date, (0, 0, 0, 0, 0, 0, 0))))

with open('sentiment_dataset.csv', 'w') as output:
    dataset = csv.writer(output, delimiter=',')
    dataset.writerow(
        [
            'date',
            'total_pos',
            'total_neg',
            'total_neu',
            'number_pos',
            'number_neg',
            'number_neu',
            'retweet_count'
        ]
    )
    for i in sorted(stats.keys()):
        dataset.writerow((i,) + stats[i])

import csv
google_data = {}
google_header = ''
with open('GOOGL.csv', encoding="utf8") as google:
    data = csv.reader(google, delimiter=',')
    first = True
    for row in data:
        if first:
            first = False
            google_header = row[1:]
            continue
        date = int(row[0].replace('-', ''))
        google_data[date] = row[1:]

with open('dataset.csv', 'w', newline='') as output:
    with open('sentiment_dataset.csv', encoding='utf8') as sentiment:
        sentiment_data = csv.reader(sentiment, delimiter=',')
        out_file = csv.writer(output, delimiter=',')
        first = True
        for row in sentiment_data:
            if first:
                out_file.writerow(row + google_header)
                first = False
                continue
            if not row:
                continue
            sentiment_date = int(row[0])
            if sentiment_date not in google_data:
                continue
            other_data = google_data[sentiment_date]
            out_file.writerow(row + other_data)


import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

import numpy as np
import pandas as pd

dataset = pd.read_csv('sentiment_dataset.csv')
x = dataset.loc[:, 'date']
pos = dataset.loc[:, 'number_pos']
neg = dataset.loc[:, 'number_neg']
retweet = dataset.loc[:, 'retweet_count']
print(dataset.loc[:, ['number_pos', 'number_neg', 'retweet_count']])
plt.plot(x, pos/retweet)
plt.plot(x, neg/retweet)
plt.legend(['pos', 'neg'])
plt.show()

