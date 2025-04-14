import pandas as pd
file_path = 'test30_augmented.csv'
full_df = pd.read_csv(file_path)
first_50k = full_df.iloc[:50000].copy()
remaining_rows = full_df.iloc[50000:].copy()

first_50k.to_csv('mqtt_dataset.csv', index=False)
remaining_rows.to_csv('remaining_records.csv', index=False)
