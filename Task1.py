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
        categorical_features = X.select_dtypes(include=['object']).columns
        numerical_features = X.select_dtypes(exclude=['object']).columns

        base_model = RandomForestRegressor(n_estimators=40, random_state=42)
        rfe = RFE(estimator=base_model, n_features_to_select=n)
        # Creating the preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(feature_range=(-1, 1)), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )

        # Creating the modeling pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('rfe', rfe),
            ('model', base_model)
        ])
        pipeline.fit(X, y)

        selected_features_mask = pipeline.named_steps['rfe'].support_
        preprocessed_X = pipeline.named_steps['preprocessor'].transform(X)
        all_feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        selected_features = all_feature_names[selected_features_mask]
        print(selected_features)
        # selected_features = X.columns[selected_features_mask]
        # print("selected features:", selected_features)
        # print("ranking of features:", selected_features_ranking)
        return pipeline
        

    def model_1_run(self):
        print("Model 1:")
        # Train the model 1 with your best hyper parameters (if have) and features on training data.
        X = self.train_data.drop('G3', axis=1)
        y = self.train_data['G3']
        for i in range(23, 24):
            pipeline = self.train(X, y, i)
            scores = cross_val_score(pipeline, X, y, cv=10, scoring='neg_mean_squared_error')
            mse_scores = -scores
            # Computing the average RMSE across all folds
            average_mse = np.mean(mse_scores)
            # Evaluate learned model on testing data, and print the results.
            print(f"{i = }\tMean squared error\t" + str(average_mse))
        return

    def model_2_run(self):
        print("--------------------\nModel 2:")
        # Train the model 2 with your best hyper parameters (if have) and features on training data.
        X = self.train_data.drop('G3', axis=1)
        y = self.train_data['G3']
        for i in range(23, 24):

            pipeline = self.train(X, y, i)
            X_test_actual = self.test_data.drop('G3', axis=1)
            y_test_actual = self.test_data['G3']

            y_test_pred = pipeline.predict(X_test_actual)
            mse = mean_squared_error(y_test_actual, y_test_pred)

            # Evaluate learned model on testing data, and print the results.
            print(f"{i = }\tMean squared error\t" + str(mse))
        return


if __name__ == "__main__":
    t1 = Task1()
    t1.model_1_run()
    t1.model_2_run()