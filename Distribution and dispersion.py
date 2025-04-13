import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('mqtt_dataset.csv')
plt.figure(figsize=(12, 8))
palette = {
    'legitimate': '#3274A1',  # blue
    'dos': '#E1812C',         # orange
    'slowite': '#3A923A',     # green
    'flood': '#C03D3E',       # red
    'bruteforce': '#9372B2',  # purple
    'malformed': '#845B53'    # brown
}

order = ['legitimate'] + sorted([t for t in df['target'].unique() if t != 'legitimate'])
ax = sns.violinplot(
    x='target', 
    y='mqtt.msgtype', 
    data=df, 
    inner='quartile',
    order=order,
    palette=palette,
    linewidth=1
)

plt.title('Distribution and Dispersion of MQTT Message Types by Attack Category', 
          fontsize=14, fontweight='bold')
plt.xlabel('Attack Category', fontsize=12, fontweight='bold')
plt.ylabel('MQTT Message Type', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.3)


for i, target in enumerate(order):
    target_data = df[df['target'] == target]['mqtt.msgtype']
    mean = target_data.mean()
    std = target_data.std()
    iqr = target_data.quantile(0.75) - target_data.quantile(0.25)
    plt.text(i, -0.7, f'Mean: {mean:.2f}', ha='center', fontsize=9, fontweight='bold')
    plt.text(i, -1.2, f'SD: {std:.2f}', ha='center', fontsize=9)
    plt.text(i, -1.7, f'IQR: {iqr:.1f}', ha='center', fontsize=9)

mqtt_types = {
    0: 'CONNECT',
    1: 'CONNACK',
    2: 'PUBLISH',
    3: 'PUBACK',
    4: 'PUBREC',
    5: 'PUBREL',
    8: 'SUBACK',
    9: 'UNSUBSCRIBE',
    12: 'PINGRESP',
    13: 'DISCONNECT'
}


y_pos = 0.5
plt.figtext(0.92, y_pos, "MQTT Message Types:", va="center", fontsize=9, fontweight='bold')
y_pos -= 0.02

for type_id in sorted(mqtt_types.keys()):
    y_pos -= 0.02
    plt.figtext(0.92, y_pos, f"{type_id}: {mqtt_types[type_id]}", va="center", fontsize=8)

plt.ylim(-2.2, 14)
plt.tight_layout(rect=[0, 0.05, 0.88, 0.98])
plt.savefig('mqtt_distribution_dispersion.png', dpi=300, bbox_inches='tight')
print("Dispersion Statistics by Attack Category:")
print("=" * 60)
print("{:<15} {:<15} {:<15} {:<15}".format('Category', 'Std Deviation', 'IQR', 'Variance'))
print("-" * 60)

for target in targets:
    target_data = df[df['target'] == target]['mqtt.msgtype']
    std_dev = target_data.std()
    iqr = target_data.quantile(0.75) - target_data.quantile(0.25)
    variance = target_data.var()
    
    print("{:<15} {:<15.4f} {:<15.4f} {:<15.4f}".format(
        target, std_dev, iqr, variance))

print("=" * 60)
print("Overall Dataset:")
overall_std = df['mqtt.msgtype'].std()
overall_iqr = df['mqtt.msgtype'].quantile(0.75) - df['mqtt.msgtype'].quantile(0.25)
overall_var = df['mqtt.msgtype'].var()
print("{:<15} {:<15.4f} {:<15.4f} {:<15.4f}".format(
    "All data", overall_std, overall_iqr, overall_var))

print("\nCoefficient of Variation (CV) by Attack Category:")
print("=" * 60)
print("{:<15} {:<15} {:<15}".format('Category', 'Mean', 'CV (%)'))
print("-" * 60)

for target in targets:
    target_data = df[df['target'] == target]['mqtt.msgtype']
    mean = target_data.mean()
    std_dev = target_data.std()
    cv = (std_dev / mean) * 100 if mean > 0 else float('inf')
    
    print("{:<15} {:<15.4f} {:<15.4f}".format(
        target, mean, cv))

print("=" * 60)
overall_mean = df['mqtt.msgtype'].mean()
overall_cv = (overall_std / overall_mean) * 100
print("{:<15} {:<15.4f} {:<15.4f}".format(
    "All data", overall_mean, overall_cv))