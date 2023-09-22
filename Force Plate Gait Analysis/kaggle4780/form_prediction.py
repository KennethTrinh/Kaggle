import pandas as pd

"""
This is a file that will help you convert your individual predictions to the final prediction. 
In the same directory as this file, you should have the following 4 files:
  - LF_test_labels.csv - with at least two columns, 'id' and 'LF'
  - LH_test_labels.csv - with at least two columns, 'id' and 'LH'
  - RF_test_labels.csv - with at least two columns, 'id' and 'RF'
  - RH_test_labels.csv - with at least two columns, 'id' and 'RH'

Running this script will convert these four files into a single CSV file, submission.csv, by
mutating the IDs so that they also include the leg that is being checked.
"""

legs = ["LF", "LH", "RF", "RH"]

dfs = []

for leg in legs:
    # read in the file
    test_prediction = pd.read_csv(f"{leg}_test_labels.csv")
    # append the abbreviation for the leg
    test_prediction['id'] = test_prediction['id'].astype(str) + f"_{leg}"
    # rename the label column
    test_prediction['label'] = test_prediction[leg]
    # exclude any potential additional columns
    dfs.append(test_prediction[['id', 'label']])

final_df = pd.concat(dfs)
final_df.to_csv("submission.csv", index=False)

