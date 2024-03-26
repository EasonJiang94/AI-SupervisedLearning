import pandas as pd

class DatasetParser(object):
    columns = [
    "school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu",
    "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures",
    "edusupport", "nursery", "higher", "internet", "romantic", "famrel",
    "freetime", "goout", "Dalc", "Walc", "health", "absences", "G3"
    ]
    
    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(path, sep="\t", header=None, names=DatasetParser.columns)


if __name__ == "__main__":
    # train = "data/assign3_students_train.txt"
    # parser = DatasetParser(train)
    # print(parser.data.info())
    test = "data/assign3_students_test.txt"
    parser = DatasetParser(test)
    print(parser.data.info())