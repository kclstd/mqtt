import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


df = pd.read_csv('mqtt_dataset.csv')

# Function to create class intervals
def create_class_intervals(data, column_name, num_classes=10):
    values = data[column_name].dropna()
    if len(values) == 0:
        return None
    
    min_val = values.min()
    max_val = values.max()
   
    width = (max_val - min_val) / num_classes
    
    # Round width to a convenient number
    magnitude = 10 ** np.floor(np.log10(width) if width > 0 else 0)
    width = np.ceil(width / magnitude) * magnitude
    
    # Create class intervals
    intervals = []
    counts = []
    
    lower = min_val
    for i in range(num_classes):
        upper = lower + width
        if i == num_classes - 1:
            # Include the maximum value in the last interval
            count = ((values >= lower) & (values <= upper)).sum()
        else:
            count = ((values >= lower) & (values < upper)).sum()
        
        intervals.append(f"{lower:.3f} - {upper:.3f}")
        counts.append(count)
        lower = upper
    
    # Create a DataFrame for the class intervals
    interval_df = pd.DataFrame({
        'Interval': intervals,
        'Frequency': counts,
        'Relative Frequency': [count / len(values) for count in counts],
        'Cumulative Frequency': np.cumsum(counts)
    })
    
    return {
        'column': column_name,
        'min': min_val,
        'max': max_val,
        'width': width,
        'num_classes': num_classes,
        'intervals': interval_df
    }

# Columns for analysis 
columns_to_analyze = [
    'tcp.len', 
    'tcp.time_delta', 
    'mqtt.len'
]

# Class intervals for each column
results = {}
for column in columns_to_analyze:
    result = create_class_intervals(df, column)
    if result:
        results[column] = result

for column, result in results.items():
    print(f"\n=== Class Intervals for {column} ===")
    print(f"Min: {result['min']}, Max: {result['max']}, Width: {result['width']}")
    print(tabulate(result['intervals'], headers='keys', tablefmt='grid', floatfmt='.3f'))
    result['intervals'].to_csv(f"{column.replace('.', '_')}_intervals.csv", index=False)

target_counts = df['target'].value_counts().reset_index()
target_counts.columns = ['Target Category', 'Frequency']
target_counts['Relative Frequency'] = target_counts['Frequency'] / len(df)
target_counts['Cumulative Frequency'] = np.cumsum(target_counts['Frequency'])

print("\n=== Distribution of Target Categories ===")
print(tabulate(target_counts, headers='keys', tablefmt='grid', floatfmt='.3f'))
target_counts.to_csv("target_distribution.csv", index=False)

# Create histograms with the class intervals
for column in columns_to_analyze:
    plt.figure(figsize=(10, 6))
    result = results[column]

    plt.hist(df[column].dropna(), bins=result['num_classes'], 
             edgecolor='black', alpha=0.7)
    
    plt.title(f'Histogram of {column} with {result["num_classes"]} Classes')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{column.replace('.', '_')}_histogram.png", dpi=300)
    plt.close()