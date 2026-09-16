import pandas as pd

## same news dataset used throughout the project, 446 rows as of Day 3's growth run
df = pd.read_csv("news_dataset.csv")
print("Rows loaded:", len(df))
print(df.head())

## headlines are single sentences, not full articles -- worth deciding before chunking:
## either fetch real article body text for a batch of these URLs to get documents
## actually worth chunking, or work with a different document source entirely
