import glob
import pandas as pd

medium_files = glob.glob("/content/Final_project/data/medium/*.csv")
reddit_files = glob.glob("/content/Final_project/data/reddit/*.csv")

print("Medium CSVs:", medium_files)
print("Reddit CSVs:", reddit_files)

dfs = []

for f in medium_files + reddit_files:
    print("Loaded:", f)
    df = pd.read_csv(f)
    dfs.append(df)