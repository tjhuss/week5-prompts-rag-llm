import pandas as pd
import json

## news_dataset.csv is the grown Week 3 dataset (278 rows), same one used since Day 5 of that week
df = pd.read_csv("news_dataset.csv")
print("Rows loaded:", len(df))
print(df["Category"].value_counts())
