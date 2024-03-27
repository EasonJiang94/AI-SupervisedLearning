# This is for INFSCI 2440 in Spring 2024
# Task 2: Multi-category task 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from utils.datasetparser import DatasetParserTask2
import warnings
from xgboost import XGBClassifier
import time
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import RandomizedSearchCV
warnings.filterwarnings('ignore')

class Task2:
    def __init__(self):
        print("================Task 2================")
        # Load the dataset
        self.train_data = DatasetParserTask2('data/assign3_students_train.txt').data
        self.test_data = DatasetParserTask2('data/assign3_students_test.txt').data
        self.categories = ["teacher", "health", "service", "at_home", "other"]
        # Prepare data
        self.X_train, self.y_train = self.prepare_data(self.train_data)
        self.X_test, self.y_test = self.prepare_data(self.test_data, training=False)
        missing_cols = set(self.X_train.columns) - set(self.X_test.columns)
        for c in missing_cols:
            self.X_test[c] = False
        self.X_test = self.X_test[self.X_train.columns]

    def prepare_data(self, data, training=True):
        # Encoding the target variable 'Mjob'
        if training:
            self.le = LabelEncoder()
            y = self.le.fit_transform(data['Mjob'])
        else:
            y = self.le.transform(data['Mjob'])
        
        X = data.drop(columns=['Mjob'])
        return X, y

    def model_1_run(self):
        print("Model 1: SVM")
        # Define the model
        model = SVC(probability=True)  # 'probability=True' allows using 'predict_proba'
        
        # Hyperparameters to tune
        param_grid = {
            'C': [0.1, 1, 10, 100],  # Regularization parameter
            'gamma': ['scale', 'auto'],  # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
            'kernel': ['rbf', 'poly', 'sigmoid']  # Specifies the kernel type to be used in the algorithm
        }
        
        # Grid search with 10-fold cross-validation
        grid_search = GridSearchCV(model, param_grid, cv=10, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)
        best_model = grid_search.best_estimator_
        
        # Evaluate the best model on the test data
        predictions = best_model.predict(self.X_test)
        self.evaluate_model(predictions, "SVM")

    def model_2_run(self):
        print("Model 2: XGBoost with SMOTE")
        # Create a pipeline that applies SMOTE and then trains an XGBoost model
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
        ])
        
        # Define hyperparameters for tuning
        param_dist = {
            'xgb__n_estimators': [100, 200, 500],
            'xgb__learning_rate': [0.01, 0.05, 0.1],
            'xgb__max_depth': [3, 5, 7, 9],
            'xgb__subsample': [0.6, 0.8, 1.0],
            'xgb__colsample_bytree': [0.6, 0.8, 1.0]
        }
        
        # Randomized search on hyperparameters
        random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=25, scoring='accuracy', cv=10, verbose=0, random_state=42)
        
        # Fit the model
        random_search.fit(self.X_train, self.y_train)
        
        # Best model
        best_model = random_search.best_estimator_
        
        # Predictions
        predictions = best_model.predict(self.X_test)
        
        # Evaluate the model
        self.evaluate_model(predictions, "XGBoost with SMOTE")

    def evaluate_model(self, predictions, model_name):
        # Compute and print the classification report
        report = classification_report(self.y_test, predictions, target_names=self.categories, output_dict=True)
        accuracy = accuracy_score(self.y_test, predictions)
        print(f"Results for {model_name}:")
        self.print_macro_results(accuracy, report['macro avg']['precision'], report['macro avg']['recall'], report['macro avg']['f1-score'])
        for category in self.categories:
            self.print_category_results(category, report[category]['precision'], report[category]['recall'], report[category]['f1-score'])

    def print_category_results(self, category, precision, recall, f1):
        print("Category\t" + category + "\tF1\t" + str(f1) + "\tPrecision\t" + str(precision) + "\tRecall\t" + str(
            recall))

    def print_macro_results(self, accuracy, precision, recall, f1):
        print("Accuracy\t" + str(accuracy) + "\tMacro_F1\t" + str(f1) + "\tMacro_Precision\t" + str(
            precision) + "\tMacro_Recall\t" + str(recall))

if __name__ == "__main__":
    task_2 = Task2()
    start_time = time.time()
    task_2.model_1_run()
    print(f"Model 1 takes {time.time() - start_time} seconds")
    start_time = time.time()
    task_2.model_2_run()
    print(f"Model 2 takes {time.time() - start_time} seconds")