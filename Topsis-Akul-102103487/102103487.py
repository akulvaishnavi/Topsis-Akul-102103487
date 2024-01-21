import sys
import pandas as pd
import numpy as np

def topsis(input_file, weights, impacts, result_file):
    try:
        data = pd.read_csv(input_file)

        if len(weights) != len(impacts) or len(weights) != data.shape[1] - 1:
            raise ValueError("Incorrect number of parameters")

        if data.shape[1] < 3:
            raise ValueError("Input file must contain three or more columns")

        if not data.iloc[:, 1:].applymap(np.isreal).all().all():
            raise ValueError("Non-numeric values found in columns 2 and beyond")

        if len(weights) != len(impacts) or len(weights) != data.shape[1] - 1:
            raise ValueError("Number of weights, impacts, and columns must be the same")

        if not all(impact in ['+', '-'] for impact in impacts):
            raise ValueError("Impacts must be either + or -")

        norm_data = data.iloc[:, 1:].apply(lambda x: x / np.linalg.norm(x), axis=0)

        weighted_norm_data = norm_data.apply(lambda x: x * np.array(weights), axis=1)

        # Calculate ideal and negative-ideal solutions
        ideal_best = weighted_norm_data.max()
        ideal_worst = weighted_norm_data.min()

        # Calculate separation measures
        separation_best = np.linalg.norm(weighted_norm_data - ideal_best, axis=1)
        separation_worst = np.linalg.norm(weighted_norm_data - ideal_worst, axis=1)

        # Calculate TOPSIS score
        topsis_score = separation_worst / (separation_best + separation_worst)

        # Add TOPSIS score to the original data
        data['TOPSIS_Score'] = topsis_score

        # Rank the alternatives
        data['Rank'] = data['TOPSIS_Score'].rank(ascending=False)

        # Save the result to a new CSV file
        data.to_csv(result_file, index=False)

        print("TOPSIS analysis completed. Results saved to", result_file)

    except FileNotFoundError:
        print("File not found:", input_file)
    except ValueError as e:
        print("Error:", str(e))
    except Exception as e:
        print("An unexpected error occurred:", str(e))

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Incorrect number of parameters. Usage: python topsis_script.py inputFileName Weights Impacts resultFileName")
    else:
        input_file = sys.argv[1]
        weights = list(map(float, sys.argv[2].split(',')))
        impacts = list(sys.argv[3])
        result_file = sys.argv[4]

        topsis(input_file, weights, impacts, result_file)