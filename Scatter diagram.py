import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr


df = pd.read_csv('mqtt_dataset.csv')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Scatter plot for tcp.time_delta vs mqtt.msgtype
jitter_amount = 0.2
x_jittered = df['mqtt.msgtype'] + np.random.uniform(-jitter_amount, jitter_amount, size=len(df))

ax1.scatter(x_jittered, df['tcp.time_delta'], alpha=0.5, s=10)
ax1.set_title('tcp.time_delta vs mqtt.msgtype')
ax1.set_xlabel('mqtt.msgtype')
ax1.set_ylabel('tcp.time_delta')
ax1.grid(True, linestyle='--', alpha=0.7)
unique_msgtypes = sorted(df['mqtt.msgtype'].unique())
ax1.set_xticks(unique_msgtypes)

#  Pearson correlation
corr1, p_value1 = pearsonr(df['mqtt.msgtype'], df['tcp.time_delta'])
ax1.text(0.05, 0.95, f'Pearson r: {corr1:.4f}\np-value: {p_value1:.4e}', 
         transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.8))
slope, intercept = np.polyfit(df['mqtt.msgtype'], df['tcp.time_delta'], 1)
x_reg = np.array([min(df['mqtt.msgtype']), max(df['mqtt.msgtype'])])
y_reg = slope * x_reg + intercept
ax1.plot(x_reg, y_reg, 'r-', linewidth=2)

# Scatter plot for mqtt.len vs mqtt.msgtype
ax2.scatter(x_jittered, df['mqtt.len'], alpha=0.5, s=10)
ax2.set_title('mqtt.len vs mqtt.msgtype')
ax2.set_xlabel('mqtt.msgtype')
ax2.set_ylabel('mqtt.len')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.set_xticks(unique_msgtypes)
corr2, p_value2 = pearsonr(df['mqtt.msgtype'], df['mqtt.len'])
ax2.text(0.05, 0.95, f'Pearson r: {corr2:.4f}\np-value: {p_value2:.4e}', 
         transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.8))


slope, intercept = np.polyfit(df['mqtt.msgtype'], df['mqtt.len'], 1)
x_reg = np.array([min(df['mqtt.msgtype']), max(df['mqtt.msgtype'])])
y_reg = slope * x_reg + intercept
ax2.plot(x_reg, y_reg, 'r-', linewidth=2)
plt.suptitle('MQTT Message Type Correlations', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig('mqtt_correlations.png', dpi=300)
plt.show()

print("\nCorrelation Analysis Report:")
print("-" * 50)
print(f"1. tcp.time_delta vs mqtt.msgtype:")
print(f"   Pearson correlation coefficient: {corr1:.6f}")
print(f"   p-value: {p_value1:.6e}")
print(f"   Interpretation: {'Strong' if abs(corr1) > 0.7 else 'Moderate' if abs(corr1) > 0.3 else 'Weak'} {'positive' if corr1 > 0 else 'negative'} correlation")
print("-" * 50)
print(f"2. mqtt.len vs mqtt.msgtype:")
print(f"   Pearson correlation coefficient: {corr2:.6f}")
print(f"   p-value: {p_value2:.6e}")
print(f"   Interpretation: {'Strong' if abs(corr2) > 0.7 else 'Moderate' if abs(corr2) > 0.3 else 'Weak'} {'positive' if corr2 > 0 else 'negative'} correlation")
print("-" * 50)

plt.figure(figsize=(16, 10))

# Boxplot for tcp.time_delta by mqtt.msgtype
plt.subplot(2, 1, 1)
sns.boxplot(x='mqtt.msgtype', y='tcp.time_delta', data=df)
plt.title('Distribution of tcp.time_delta by mqtt.msgtype')
plt.xlabel('mqtt.msgtype')
plt.ylabel('tcp.time_delta')
plt.grid(True, linestyle='--', alpha=0.7)

# Boxplot for mqtt.len by mqtt.msgtype
plt.subplot(2, 1, 2)
sns.boxplot(x='mqtt.msgtype', y='mqtt.len', data=df)
plt.title('Distribution of mqtt.len by mqtt.msgtype')
plt.xlabel('mqtt.msgtype')
plt.ylabel('mqtt.len')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('mqtt_boxplots.png', dpi=300)
plt.show()