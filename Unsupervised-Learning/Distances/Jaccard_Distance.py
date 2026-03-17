import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import jaccard
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('/workspaces/Supervised-Machine-Learning/Datasets/Real estate.csv')
X1 = data["X5 latitude"]
X2 = data["X6 longitude"]

X1 = StandardScaler().fit_transform(X1.values.reshape(-1, 1)).flatten()
X2 = StandardScaler().fit_transform(X2.values.reshape(-1, 1)).flatten()

distance_12 = jaccard(X1, X2)

print(f"Jaccard Distance between X1 and X2: {distance_12}")