# This is for INFSCI 2440 in Spring 2024
# Task 2: Multi-category task 

class Task2:
    # please feel free to create new python files, adding functions and attributes to do training, validation, testing

    def __init__(self):
        print("================Task 2================")
        return

    def print_category_results(self, category, precision, recall, f1):
        print("Category\t" + category + "\tF1\t" + str(f1) + "\tPrecision\t" + str(precision) + "\tRecall\t" + str(
            recall))

    def print_macro_results(self, accuracy, precision, recall, f1):
        print("Accuracy\t" + str(accuracy) + "\tMacro_F1\t" + str(f1) + "\tMacro_Precision\t" + str(
            precision) + "\tMacro_Recall\t" + str(recall))

    def model_1_run(self):
        print("Model 1:")
        # Train the model 1 with your best hyper parameters (if have) and features on training data.


        # Evaluate learned model on testing data, and print the results.
        self.print_macro_results(0.0, 0.0, 0.0, 0.0)
        categories = ["teacher", "health", "service", "at_home", "other"]
        for category in categories:
            self.print_category_results(category, 0.0, 0.0, 0.0)
        return

    def model_2_run(self):
        print("--------------------\nModel 2:")
        # Train the model 2 with your best hyper parameters (if have) and features on training data.


        # Evaluate learned model on testing data, and print the results.
        self.print_macro_results(0.0, 0.0, 0.0, 0.0)
        categories = ["teacher", "health", "service", "at_home", "other"]
        for category in categories:
            self.print_category_results(category, 0.0, 0.0, 0.0)
        return
