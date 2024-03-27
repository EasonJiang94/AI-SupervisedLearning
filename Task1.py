# This is for INFSCI 2440 in Spring 2024
# Task 1: Regression task 
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np
from utils.datasetparser import DatasetParser


class Task1:
    # please feel free to create new python files, adding functions and attributes to do training, validation, testing
    TRAIN_SET_PATH = "data/assign3_students_train.txt"
    TEST_SET_PATH = "data/assign3_students_test.txt"
    def __init__(self):
        print("================Task 1================")
        self.train_data = DatasetParser(Task1.TRAIN_SET_PATH).data
        self.test_data = DatasetParser(Task1.TEST_SET_PATH).data        
        
    
    def train(self, X, y, n=27):
        # Identifying categorical columns to be one-hot encoded

        base_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rfe = RFE(estimator=base_model, n_features_to_select=n)
        rfe.fit(X, y)
        selected_features = X.columns[rfe.support_]
        #print("Selected features:", selected_features)
        X_selected = X[selected_features]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_selected, y)
        return model, selected_features
        

    def model_1_run(self):
        print("Model 1:")
        # Train the model 1 with your best hyper parameters (if have) and features on training data.
        X = self.train_data.drop('G3', axis=1)
        y = self.train_data['G3']
        min_avg_mse = 99999999
        self.optimized_feature_number = -1
        self.optimized_features = None
        for i in range(1, len(X.columns)+1):
            model, selected_features = self.train(X, y, i)
            scores = cross_val_score(model, X[selected_features], y, cv=10, scoring='neg_mean_squared_error')
            mse_scores = -scores
            # Computing the average RMSE across all folds
            average_mse = np.mean(mse_scores*400)
            # Evaluate learned model on testing data, and print the results.
            if average_mse < min_avg_mse:
                min_avg_mse = average_mse
                self.optimized_feature_number = i
                self.optimized_features = selected_features
            print(f"{i = }\tMean squared error\t" + str(average_mse))
        print(f"Mean squared error\t" + str(min_avg_mse))
        print(f"{self.optimized_feature_number = }")
        print(f"{self.optimized_features = }")
        return

    def model_2_run(self):
        print("--------------------\nModel 2:")
        # Train the model 2 with your best hyper parameters (if have) and features on training data.
        X = self.train_data.drop('G3', axis=1)
        y = self.train_data['G3']
        model, selected_featrures = self.train(X[self.optimized_features], y, self.optimized_feature_number)
        X_test_actual = self.test_data.drop('G3', axis=1)
        y_test_actual = self.test_data['G3']
        for feature in list(selected_featrures):
            if feature not in X_test_actual.columns:
                X_test_actual[feature] = False
        y_test_pred = model.predict(X_test_actual[selected_featrures])
        mse = mean_squared_error(y_test_actual*20, y_test_pred*20)

        # Evaluate learned model on testing data, and print the results.
        print(f"Mean squared error\t" + str(mse))
        return


if __name__ == "__main__":
    t1 = Task1()
    t1.model_1_run()
    t1.model_2_run()