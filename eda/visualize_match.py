import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd

import sys

opp_name = sys.argv[1]
idx = sys.argv[2]

df = pd.read_csv(f"data/{opp_name}.txt", comment="#") 
indices = [i for i in range(1, df.shape[0])
           if df.iloc[i]['hand_number'] != df.iloc[i-1]['hand_number']]
rewards = df.iloc[indices][f'team_{idx}_bankroll'].diff().dropna()

sn.displot(rewards, binwidth=5, aspect=2)
plt.savefig(f"plots/{opp_name}.png") 