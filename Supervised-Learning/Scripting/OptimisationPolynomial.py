import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import threading

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("/workspaces/Supervised-Machine-Learning/Datasets/Real estate.csv")
print("The dataframe description is: ")
print(data.describe())

X = data['X5 latitude']
y = data.iloc[:, -1]

class PolynomialRegression(threading.Thread):
    def __init__(self, degree, X, y):
        super().__init__()
        self.y_pred = None
        self.X_test_poly = None
        self.poly_features = None
        self.X_train_poly = None
        self.X_test = None
        self.X_train = None
        self.y_test = None
        self.y_train = None
        self.degree = degree
        self.model = LinearRegression()
        self.data_split(X, y, 0.2)

    def data_split(self, X: pd.Series , y: pd.Series, r: float) -> None:
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=r, random_state=42)
        
    def fit(self) -> None:
        self.poly_features = PolynomialFeatures(degree=self.degree)
        self.X_train_poly = self.poly_features.fit_transform(self.X_train)
        self.X_test_poly = self.poly_features.transform(self.X_test)
        self.model.fit(self.X_train_poly, self.y_train)
    
    def Evaluation(self) -> None:
        self.y_pred = self.model.predict(self.X_test_poly)
        self.rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        self.r2 = r2_score(self.y_test, self.y_pred)
        self.mae = mean_absolute_error(self.y_test, self.y_pred)
        
    def plot(self) -> None:
        # Create a figure with subplots for training and test data
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Sort data for smooth plotting
        X_train_sorted = np.sort(self.X_train.values)
        X_train_poly_sorted = self.poly_features.transform(X_train_sorted.reshape(-1, 1))
        y_train_pred_sorted = self.model.predict(X_train_poly_sorted)

        # Plot Training Data
        axes[0].scatter(self.X_train, self.y_train, color='blue', label='Actual', alpha=0.6)
        axes[0].plot(X_train_sorted, y_train_pred_sorted, color='red', linewidth=2, label='Predicted')
        axes[0].set_xlabel('Number of Convenience Stores')
        axes[0].set_ylabel('House Price of Unit Area')
        axes[0].set_title('Training Data: Actual vs Predicted')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot Test Data
        axes[1].scatter(self.X_test, self.y_test, color='green', label='Actual', alpha=0.6)
        axes[1].scatter(self.X_test, self.y_pred, color='red', marker='x', s=100, label='Predicted', alpha=0.8)
        axes[1].set_xlabel('Number of Convenience Stores')
        axes[1].set_ylabel('House Price of Unit Area')
        axes[1].set_title('Test Data: Actual vs Predicted')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.show()





