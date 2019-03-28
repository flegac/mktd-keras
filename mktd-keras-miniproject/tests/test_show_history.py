import pandas

from exercices.visualize import show_history

history = pandas.read_csv('../project/training/training_logs.csv')
show_history(history)
