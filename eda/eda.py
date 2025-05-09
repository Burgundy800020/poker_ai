from matplotlib import ticker
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay, calibration_curve

# street0   = np.loadtxt(f"data/street_{0}_equity.txt", delimiter=",")
# street1   = np.loadtxt(f"data/street_{1}_equity.txt", delimiter=",")

# print(np.abs(street0-street1).sum())

win_true = np.loadtxt("data/win_true.txt", delimiter=',')
for i in range(4):
    equities = np.loadtxt(f"data/street_{i}_equity.txt", delimiter=",")
    print(f"street {i}: {np.quantile(equities, 0.65)}")
    g = sn.displot(equities, bins=40, aspect=2.5)
    # Set tick graduation (intervals)
    for ax in g.axes.flat:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))  # X-axis tick interval of 0.5
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))  # Y-axis tick interval of 0.1
    plt.title(f"street {i}")
    plt.subplots_adjust(top=0.9)
    plt.savefig(f"plots/street_{i}_equity.png") 
    plt.clf()
    n_bins = 80 if i == 0 else 40
    prob_true, prob_pred = calibration_curve(win_true, equities, n_bins=n_bins)
    disp = CalibrationDisplay(prob_true, prob_pred, equities)
    disp.plot()
    plt.title(f"street {i}")
    plt.subplots_adjust(top=0.9)
    plt.savefig(f'plots/street_{i}_calib.png')


