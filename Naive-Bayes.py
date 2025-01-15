import os
import sys
import pandas as pd

class FileManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.changes = {}

    def load_excel(self):
        if os.path.exists(self.file_path):
            return pd.read_excel(self.file_path)
        else:
            raise FileNotFoundError(f"No file found at {self.file_path}")

    def get_training_data(self):
        data = self.load_excel()
        training_data = data.dropna(subset=[data.columns[-1]])
        return training_data

    def get_prediction_data(self):
        data = self.load_excel()
        prediction_data = data[data[data.columns[-1]].isna()]
        return prediction_data
    
    def create_output_excel(self):
        data = self.load_excel()
        output_file_path = os.path.splitext(self.file_path)[0] + "_completed.xlsx"
        data.to_excel(output_file_path, index=False)
        print(f"Output file created at {output_file_path}")

    def add_to_output_excel(self, row, column, value):
        self.changes[(row, column)] = value

    def save_to_output_excel(self, row, column, value):
        output_file_path = os.path.splitext(self.file_path)[0] + "_completed.xlsx"
        data = self.load_excel()
        for (row, column), value in self.changes.items():
            data.at[row, column] = value
        data.to_excel(output_file_path, index=False)
        print(f"Output file updated at {output_file_path}")

class NaiveBayes:
    def __init__(self):
        self.file_manager = None
        self.probabilities = {}
        self.columns_map = {}
        self.last_column = None

    def run_on_excel(self, excel_file_path):
        self.file_manager = FileManager(excel_file_path)
        try:

            self.train(self.file_manager.get_training_data())

            self.predict(self.file_manager.get_prediction_data())

        except FileNotFoundError as e:
            print(e)

    def train(self, training_data):
        if __debug__:
            print("********** Training data **********" )
        columns = len(training_data.columns)
        rows = len(training_data)
        self.last_column = training_data.columns[-1]

        for column in training_data.columns:
            self.columns_map[column] = training_data[column].unique()

        if __debug__:
            print("Columns map:")
            for column in self.columns_map:
                print(column)
                print(self.columns_map[column])

        for index, row in training_data.iterrows():
            result = row[self.last_column]
            if result not in self.probabilities:
                self.probabilities[result] = {}
                self.probabilities[result]["internal_count"] = 0
                for column in self.columns_map:
                    self.probabilities[result][column] = {}
                    for value in self.columns_map[column]:
                        self.probabilities[result][column][value] = 1 # Laplace smoothing
            
            self.probabilities[result]["internal_count"] += 1

            for column in self.columns_map:
                value = row[column]
                self.probabilities[result][column][value] += 1

        if __debug__:
            print("Probabilities structure:")
            for result in self.probabilities:
                print(result)
                for column in self.probabilities[result]:
                    if column == "internal_count":
                        print(f"  internal_count: {self.probabilities[result][column]}")
                        continue
                    print(f"  {column}")
                    for value in self.probabilities[result][column]:
                        print(f"    {value}: {self.probabilities[result][column][value]}")


    def predict(self, prediction_data):
        if __debug__:
            print("********** Prediction data **********" )
        
        self.file_manager.create_output_excel()

        for index, row in prediction_data.iterrows():
            chances = self.get_chances_dict()
            for chance in chances:
                for column in self.columns_map:
                    if column == self.last_column:
                        continue
                    value = row[column]
                    chances[chance] *= self.probabilities[chance][column][value] / self.probabilities[chance]["internal_count"]
            chances = self.get_normalized_chances(chances)

            chance_key, chance_value = self.get_maximum_chance(chances)

            row_string = ', '.join(f'{col}: {val}' for col, val in row.items())
            print(f"Prediction for row {index}: [{row_string}]")
            for chance in chances:
                if chance == chance_key:
                    print(f"\033[92m  {chance}: {chances[chance]}\033[0m")  # Green color
                else:
                    print(f"\033[91m  {chance}: {chances[chance]}\033[0m")  # Red color
            self.file_manager.add_to_output_excel(index, self.last_column, chance_key)
            
        self.file_manager.save_to_output_excel(index, self.last_column, chance_key)

    def get_chances_dict(self):
        chances = {}
        for value in self.columns_map[self.last_column]:
            chances[value] = 1
        return chances
    
    def get_normalized_chances(self, chances):
        total = sum(chances.values())
        for value in chances:
            chances[value] /= total
        return chances
    
    def get_maximum_chance(self, chances):
        best_chance_value = 0
        for chance in chances:
            if chances[chance] > best_chance_value:
                best_chance = chance
                best_chance_value = chances[chance]
        return best_chance, best_chance_value

def main():
    if len(sys.argv) != 2:
        print("Usage: python Naive-Bayes.py path/to/your/excel_file.xlsx")
        return
    
    excel_file_path = sys.argv[1]
    alg = NaiveBayes()
    alg.run_on_excel(excel_file_path)

if __name__ == "__main__":
    main()