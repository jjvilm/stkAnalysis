import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
import pandas as pd
from matplotlib import style
#style.use("ggplot")
style.use("dark_background")

def Build_Data_Set(features = ["DE Ratio",
                                "Trailing P/E"]):
    data_df = pd.DataFrame.from_csv("key_stats.csv")
    X = np.array(data_df[features].values)#.tolist())

    y = (data_df["Status"]
        .replace("underperformed",0)
        .replace("outperformed",1)
        .values.tolist())
    # helps preprocess data before using for training
    X = preprocessing.scale(X)

    return X,y 

def Analysis():
    X, y = Build_Data_Set()

    clf = svm.SVC(kernel="linear", C=1.0)
    clf.fit(X,y)
    # since graphics, following needs to happen to see visuals
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min(X[:,0]),max(X[:,0]))
    yy = a * xx - clf.intercept_[0] / w[1]
    ########################################

    h0 = plt.plot(xx,yy,"k-", label="non weighted")
    plt.scatter(X[:,0], X[:,1])
    plt.ylabel("Trailing P/E")
    plt.xlabel("DE Ratio")
    plt.legend()

    plt.show()

Analysis()

