import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def sigmoid(x):
    return 3*(2 / (1 + np.exp(-x)) - 1)

class DatasetParser(object):
    columns = [
        "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
        "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures",
        "edusupport", "nursery", "higher", "internet", "romantic", "famrel",
        "freetime", "goout", "Dalc", "Walc", "health", "absences", "G3"
    ]
    
    binary_columns = [
        "school", "sex", "address", "famsize", "Pstatus", "nursery", "higher", "internet", "romantic"
    ]
    non_binary_categorical_columns = [
        "Mjob", "Fjob", "reason", "guardian", "edusupport"  
    ]
    range_info = {
        "age" : {"min" : 15, "max": 22},
        "Medu" : {"min" : 0, "max": 4},
        "Fedu" : {"min" : 0, "max": 4},
        "traveltime" : {"min" : 1, "max": 4},
        "studytime" : {"min" : 1, "max": 4},
        "failures" : {"min" : 1, "max": 4},
        "famrel" : {"min" : 1, "max": 5},
        "freetime" : {"min" : 1, "max": 5},
        "goout" : {"min" : 1, "max": 5},
        "Dalc" : {"min" : 1, "max": 5},
        "Walc" : {"min" : 1, "max": 5},
        "health" : {"min" : 1, "max": 5},
        "absences" : {"min" : 0, "max": 93},
        "G3" : {"min" : 0, "max": 20}
    }
    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(path, sep="\t", header=None, names=DatasetParser.columns)
        self._convert_binary_to_boolean()
        self._apply_one_hot_encoding()
        self._normalize_data()
        self._feature_interaction_and_polynomial_features()
    
    def _convert_binary_to_boolean(self):
        for column in self.binary_columns:
            if column in ['school', 'sex', 'address', 'famsize', 'Pstatus']:
                # Assuming 'GP', 'M', 'U', 'LE3', 'T' as True and the opposite values as False
                self.data[column] = self.data[column].map({'GP': True, 'MS': False, 'M': True, 'F': False, 'U': True, 'R': False, 'LE3': True, 'GT3': False, 'T': True, 'A': False})
            else:
                # For 'nursery', 'higher', 'internet', 'romantic' assuming 'yes' as True and 'no' as False
                self.data[column] = self.data[column].map({'yes': True, 'no': False})

    def _apply_one_hot_encoding(self):
        self.data = pd.get_dummies(self.data, columns=self.non_binary_categorical_columns)

    def _normalize_data(self):
        # reverse projection age from 22 to 15 (0 to 1)
        self.data['age'] = self.data['age'].apply(lambda x: -1 + 2 * (22 - x) / (22 - 15))
        
        # Sigmoid
        education_features = ['Medu', 'Fedu', 'traveltime', 'studytime', 'failures',
                              'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health']
        for feature in education_features:
            self.data[feature] = sigmoid((self.data[feature] - DatasetParser.range_info[feature]["min"]) / 
                                         (DatasetParser.range_info[feature]["max"] - DatasetParser.range_info[feature]["min"]))
        
        self.data['absences'] = (self.data['absences'] - DatasetParser.range_info['absences']["min"]) / (DatasetParser.range_info['absences']["max"] - DatasetParser.range_info['absences']["min"])
        
        self.data['G3'] = (self.data['G3'] - DatasetParser.range_info['G3']["min"]) / (DatasetParser.range_info['G3']["max"] - DatasetParser.range_info['G3']["min"])
    
    def _feature_interaction_and_polynomial_features(self):
        self.data['parents_education'] = self.data['Medu'] * self.data['Fedu']
        self.data['famrel_freetime'] = self.data['famrel'] * self.data['freetime']
        
        continuous_features = ['age', 'traveltime', 'studytime', 'absences']
        
        pf = PolynomialFeatures(degree=2, include_bias=False)
        

        continuous_data = self.data[continuous_features]
        poly_features = pf.fit_transform(continuous_data)
        
        poly_features = np.delete(poly_features, [i for i in range(len(continuous_features))], axis=1)
        
        for i in range(poly_features.shape[1]):
            self.data[f'poly_{i}'] = poly_features[:, i]

        features_to_remove = ['Medu', 'Fedu', 'famrel', 'freetime'] 
        self.data.drop(columns=features_to_remove, inplace=True)

class DatasetParserTask2(DatasetParser):
    non_binary_categorical_columns = [
        "Fjob", "reason", "guardian", "edusupport"  
    ]
    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(path, sep="\t", header=None, names=DatasetParser.columns)
        # self.data.drop(columns=['Mjob'], inplace=True)
        self._convert_binary_to_boolean()
        self._apply_one_hot_encoding()
        self._normalize_data()
        self._feature_interaction_and_polynomial_features()
    
    def _feature_interaction_and_polynomial_features(self):
        self.data['famrel_freetime'] = self.data['famrel'] * self.data['freetime']
        
        continuous_features = ['age', 'traveltime', 'studytime', 'absences']
        pf = PolynomialFeatures(degree=2, include_bias=False)
        

        continuous_data = self.data[continuous_features]
        poly_features = pf.fit_transform(continuous_data)
        
        poly_features = np.delete(poly_features, [i for i in range(len(continuous_features))], axis=1)
        
        for i in range(poly_features.shape[1]):
            self.data[f'poly_{i}'] = poly_features[:, i]

        features_to_remove = ['Medu', 'famrel', 'freetime'] 
        self.data.drop(columns=features_to_remove, inplace=True)
    
    def _remove_useless_features(self):
        features_to_remove = [] 
        self.data.drop(columns=features_to_remove, inplace=True)

class DatasetParserTask3(DatasetParser):
    
    non_binary_categorical_columns = [
        "Mjob", "Fjob", "reason", "guardian"  
    ]
    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(path, sep="\t", header=None, names=DatasetParser.columns)
        # self.data.drop(columns=['Mjob'], inplace=True)
        self._convert_binary_to_boolean()
        self._apply_one_hot_encoding()
        self._normalize_data()
        self._feature_interaction_and_polynomial_features()

    def _feature_interaction_and_polynomial_features(self):
        # self.data['famrel_freetime'] = self.data['famrel'] * self.data['freetime']
        
        continuous_features = ['goout', 'Dalc', 'Walc', 'freetime']
        pf = PolynomialFeatures(degree=3, include_bias=False)
        

        continuous_data = self.data[continuous_features]
        poly_features = pf.fit_transform(continuous_data)
        
        poly_features = np.delete(poly_features, [i for i in range(len(continuous_features))], axis=1)
        
        for i in range(poly_features.shape[1]):
            self.data[f'Poly_goout_Dalc_Walc_{i}'] = poly_features[:, i]

        # continuous_features = ['Fedu', 'Medu', 'famrel', ]
        # pf = PolynomialFeatures(degree=3, include_bias=True)
        

        # continuous_data = self.data[continuous_features]
        # poly_features = pf.fit_transform(continuous_data)
        
        # poly_features = np.delete(poly_features, [i for i in range(len(continuous_features))], axis=1)
        
        # for i in range(poly_features.shape[1]):
        #     self.data[f'Fedu_Medu_famrel_freetime_{i}'] = poly_features[:, i]

        features_to_remove = ['Medu', 'famrel', 'freetime'] 
        self.data.drop(columns=features_to_remove, inplace=True)

if __name__ == "__main__":
    # train = "data/assign3_students_train.txt"
    # parser = DatasetParser(train)
    # print(parser.data.info())
    test = "data/assign3_students_test.txt"
    parser = DatasetParserTask2(test)
    print(parser.data.info())
    # train = "data/assign3_students_train.txt"
    # test = "data/assign3_students_test.txt"
    # parser = DatasetParser(train)
    # print(parser.data[['school', 'sex', 'address', 'famsize', 'Pstatus', 'nursery', 'higher', 'internet', 'romantic']].head())
