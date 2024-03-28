# This is for INFSCI 2440 in Spring 2024
# Task 3: Multi-label task 
from utils.datasetparser import DatasetParserTask3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer


class Task3:
    # please feel free to create new python files, adding functions and attributes to do training, validation, testing

    def __init__(self):
        print("================Task 3================")
        self.train_data = DatasetParserTask3('data/assign3_students_train.txt').data
        self.test_data = DatasetParserTask3('data/assign3_students_test.txt').data
        self.X_train, self.y_train = self.prepare_data(self.train_data)
        self.X_test, self.y_test = self.prepare_data(self.test_data, training=False)
        missing_cols = set(self.X_train.columns) - set(self.X_test.columns)
        for c in missing_cols:
            self.X_test[c] = False
        self.X_test = self.X_test[self.X_train.columns]
        return

    def prepare_data(self, data, training=True):
        import pandas as pd
        # Assuming 'data' is your DataFrame
        
        # Identify categorical columns that need encoding
        # Replace 'categorical_columns' with your actual columns
        categorical_columns = ['edusupport']
        # Exclude 'edusupport' from one-hot encoding process if it's your label
        
        # One-hot encode categorical variables
        data = pd.get_dummies(data, columns=categorical_columns[:-1])  # Exclude label column for encoding
        
        # Split data into features and labels
        X = data.drop('edusupport', axis=1)  # Exclude label column
        y = data['edusupport'].astype(str)  # Ensure label column is string
        
        # Split the string into a list of categories for labels
        y = y.str.split(' ')
        if training:
            self.mlb = MultiLabelBinarizer()
            y = self.mlb.fit_transform(y)
        else:
            y = self.mlb.transform(y)
        return X, y

    def model_1_run(self):
        print("--------------------\nModel 1: Logistic Regression with One-vs-Rest")
        model = MultiOutputClassifier(LogisticRegression(max_iter=1000))
        model.fit(self.X_train, self.y_train)

        predictions = model.predict(self.X_test)

        acc = accuracy_score(self.y_test, predictions)
        
        
        h_loss = hamming_loss(self.y_test, predictions)

        print(f"Accuracy: {acc}\tHamming loss: {h_loss}")

    def model_2_run(self):
        print("--------------------\nModel 2: Decision Tree with One-vs-Rest")

        model = MultiOutputClassifier(DecisionTreeClassifier())
        model.fit(self.X_train, self.y_train)

        predictions = model.predict(self.X_test)

        acc = accuracy_score(self.y_test, predictions)
        h_loss = hamming_loss(self.y_test, predictions)

        print(f"Accuracy: {acc}\tHamming loss: {h_loss}")

if __name__ == "__main__":
    import time
    t3 = Task3()
    start_time = time.time()
    t3.model_1_run()
    print(f"Model 1 takes {time.time() - start_time} seconds")
    start_time = time.time()
    t3.model_2_run()
    print(f"Model 2 takes {time.time() - start_time} seconds")