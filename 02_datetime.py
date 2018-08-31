import datetime
import numpy as np
import matplotlib.pyplot as plt

def datestr2num(s):
    return datetime.datetime.strptime(s, "%d-%m-%Y").date().weekday()

dates, close = np.loadtxt('data/data.csv', delimiter=',', usecols=(1,6), 
    converters={1: datestr2num}, unpack=True, encoding='utf-8')

averages = np.zeros(5)
# 计算周一到周五每天的平均收盘价
for i in range(5):
    indices = np.where(dates == i)
    prices = np.take(close, indices)
    avg = np.mean(prices)
    print("Day", i, "prices", prices, "Average", avg)
    averages[i] = avg

top = np.max(averages)
print("Highest average", top)
print("Top day of the week", np.argmax(averages))

bottom = np.min(averages)
print("Lowest average", bottom)
print("Bottom day of the week", np.argmin(averages))

# 生成等间隔的日期数组
date_range = np.arange('2015-04-22', '2015-05-22', 7, dtype='datetime64')
print(date_range)
