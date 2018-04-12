import pandas as pd

old = pd.read_csv("curtis_submission.csv")
old['label'] = 1 - old['label']
old.to_csv("curtis_submit_best.csv")
