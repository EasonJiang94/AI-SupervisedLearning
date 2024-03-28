# This is for INFSCI 2440 in Spring 2024
# Task 1: Regression task 
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np
from utils.datasetparser import DatasetParser
import time

class Task1:
    # please feel free to create new python files, adding functions and attributes to do training, validation, testing
    TRAIN_SET_PATH = "data/assign3_students_train.txt"
    TEST_SET_PATH = "data/assign3_students_test.txt"
    def __init__(self):
        print("================Task 1================")
        self.train_data = DatasetParser(Task1.TRAIN_SET_PATH).data
        self.test_data = DatasetParser(Task1.TEST_SET_PATH).data        
        self.optimized_feature_number = 37
        missing_cols = set(self.train_data.columns) - set(self.test_data.columns)
        for c in missing_cols:
            self.test_data[c] = False
        self.test_data = self.test_data[self.train_data.columns]
    
    def train(self, X, y, n=27):
        # Identifying categorical columns to be one-hot encoded

        base_model = RandomForestRegressor(n_estimators=150, random_state=42)
        rfe = RFE(estimator=base_model, n_features_to_select=n)
        rfe.fit(X, y)
        selected_features = X.columns[rfe.support_]
        #print("Selected features:", selected_features)
        X_selected = X[selected_features]
        model = RandomForestRegressor(n_estimators=150, random_state=42)
        model.fit(X_selected, y)
        return model, selected_features
        

    def model_1_run(self):
        print("Model 1:")
        # Train the model 1 with your best hyper parameters (if have) and features on training data.
        X = self.train_data.drop('G3', axis=1)
        y = self.train_data['G3']
        min_avg_mse = 99999999
        # self.optimized_feature_number = -1
        self.optimized_features = None
        for i in range(self.optimized_feature_number, self.optimized_feature_number + 1):
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
            # print(f"{i = }\tMean squared error\t" + str(average_mse))
        print(f"Average validation Mean squared error\t" + str(min_avg_mse))
        # print(f"{self.optimized_feature_number = }")
        # print(f"{self.optimized_features = }")
        self.model_1_test()
        return

    def model_1_test(self):
        print("--------------------\nTest for Model 1:")
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
    
    def train_svm_and_evaluate(self, X_train, y_train):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR())
        ])
        
        parameters = {
            'svr__C': [0.1, 1, 10],
            'svr__epsilon': [0.01, 0.1, 0.5],
            'svr__kernel': ['rbf', 'linear', 'poly'],
            'svr__gamma': ['scale', 'auto']
        }
        
        grid_search = GridSearchCV(pipeline, parameters, cv=10, scoring='neg_mean_squared_error', verbose=1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best average MSE from CV: {-grid_search.best_score_*400}")

        return best_model

    def model_2_run(self):
        # Prepare your data
        X_train = self.train_data.drop('G3', axis=1)
        y_train = self.train_data['G3']
        X_test = self.test_data.drop('G3', axis=1)
        y_test = self.test_data['G3']

        # Train and evaluate the model
        best_svm_model = self.train_svm_and_evaluate(X_train, y_train)
        
        # Make predictions on the test set
        y_pred = best_svm_model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_pred)
        print(f"Test MSE for best SVM model: {test_mse*400}")


if __name__ == "__main__":
    t1 = Task1()
    start_time = time.time()
    t1.model_1_run()
    print(f"Model 1 takes {time.time() - start_time} seconds")
    start_time = time.time()
    t1.model_2_run()
    print(f"Model 2 takes {time.time() - start_time} seconds")