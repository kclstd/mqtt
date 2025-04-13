import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


df = pd.read_csv('mqtt_dataset.csv')

# Calculate standard deviation for the entire dataset
std_dev = df['mqtt.msgtype'].std()
print(f"Standard Deviation of mqtt.msgtype: {std_dev:.4f}")

# Calculate interquartile range (IQR)
q1 = df['mqtt.msgtype'].quantile(0.25)
q3 = df['mqtt.msgtype'].quantile(0.75)
iqr = q3 - q1
print(f"Interquartile Range (IQR) of mqtt.msgtype: {iqr}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
sns.boxplot(x='target', y='mqtt.msgtype', data=df, ax=ax1)
ax1.set_title('MQTT Message Types by Attack Category')
ax1.set_xlabel('Attack Category')
ax1.set_ylabel('MQTT Message Type')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
mqtt_message_types = {
    0: 'CONNECT',
    1: 'CONNACK',
    2: 'PUBLISH',
    3: 'PUBACK',
    4: 'PUBREC',
    5: 'PUBREL',
    6: 'PUBCOMP',
    7: 'SUBSCRIBE',
    8: 'SUBACK',
    9: 'UNSUBSCRIBE',
    10: 'UNSUBACK',
    11: 'PINGREQ',
    12: 'PINGRESP',
    13: 'DISCONNECT'
}

message_counts = df['mqtt.msgtype'].value_counts().sort_index()
message_labels = [mqtt_message_types.get(i, f"Type {i}") for i in message_counts.index]

ax2.bar(message_labels, message_counts.values)
ax2.set_title('Distribution of MQTT Message Types')
ax2.set_xlabel('MQTT Message Type')
ax2.set_ylabel('Count')
ax2.set_xticklabels(message_labels, rotation=45, ha='right')


print("\nStatistics by target class:")
for target in df['target'].unique():
    subset = df[df['target'] == target]['mqtt.msgtype']
    print(f"\nTarget: {target}")
    print(f"Count: {len(subset)}")
    print(f"Mean: {subset.mean():.4f}")
    print(f"Standard Deviation: {subset.std():.4f}")
    print(f"IQR: {subset.quantile(0.75) - subset.quantile(0.25)}")
    print(f"Message type distribution:")
    print(subset.value_counts().sort_index().to_dict())

plt.tight_layout()
plt.savefig('mqtt_msgtype_analysis.png', dpi=300)
plt.show()

print("\nMQTT Message Type Patterns by Attack:")
for target in df['target'].unique():
    target_data = df[df['target'] == target]
    top_types = target_data['mqtt.msgtype'].value_counts().nlargest(3)
    print(f"\n{target.upper()} attacks:")
    for msg_type, count in top_types.items():
        type_name = mqtt_message_types.get(msg_type, f"Type {msg_type}")
        percentage = (count / len(target_data)) * 100
        print(f"  {type_name}: {count} ({percentage:.1f}%)")