import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
target_colors = ['#2C73D2', '#845EC2', '#FF6F91', '#FFC75F', '#F9F871', '#00C9A7']
tcp_flags_colors = ['#0081CF', '#845EC2', '#D65DB1', '#FF9671', '#FFC75F', '#F9F871', '#00C9A7', '#008F7A']


df = pd.read_csv('mqtt_dataset.csv')
plt.figure(figsize=(12, 10))
target_counts = df['target'].value_counts()
explode = [0.05 if x == target_counts.index[0] else 0.02 for x in target_counts.index]

# Pie chart for target distribution
wedges, texts, autotexts = plt.pie(
    target_counts, 
    labels=target_counts.index,
    autopct='%1.1f%%',
    startangle=90, 
    explode=explode,
    colors=target_colors,
    textprops={'fontsize': 14, 'fontweight': 'bold'},
    pctdistance=0.80,  
    labeldistance=1.05  
)


for text in texts:
    text.set_fontsize(14)
    text.set_fontweight('bold')
    

for autotext in autotexts:
    autotext.set_fontsize(13)
    autotext.set_fontweight('bold')
    autotext.set_color('white') 

legend_labels = [f'{label}: {count:,}' for label, count in zip(target_counts.index, target_counts.values)]
plt.legend(wedges, legend_labels, title="Counts", loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=12)

plt.title('Distribution of Target Classes in MQTT Dataset', fontsize=20, pad=20)
plt.tight_layout()
plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Bar chart for tcp.flags distribution
plt.figure(figsize=(14, 8))
tcp_flags_counts = df['tcp.flags'].value_counts().sort_values(ascending=False)

bars = plt.bar(
    tcp_flags_counts.index, 
    tcp_flags_counts.values,
    color=tcp_flags_colors[:len(tcp_flags_counts)],
    edgecolor='black',
    linewidth=1.5,
    alpha=0.8
)

for i, bar in enumerate(bars):
    bar.set_alpha(0.7 + (0.3 * (len(bars) - i) / len(bars)))

for i, count in enumerate(tcp_flags_counts.values):
    plt.text(i, count + (max(tcp_flags_counts.values) * 0.01), 
             f'{count:,}', 
             ha='center', 
             fontsize=14,
             fontweight='bold')

plt.title('Distribution of TCP Flags in MQTT Dataset', fontsize=20, pad=20)
plt.xlabel('TCP Flags Value', fontsize=16)
plt.ylabel('Count', fontsize=16)
plt.xticks(range(len(tcp_flags_counts.index)), tcp_flags_counts.index, rotation=45, fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('tcp_flags_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
