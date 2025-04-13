import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
import numpy as np


df = pd.read_csv('mqtt_dataset.csv')

# Calculate central tendency and mean for mqtt.msgtype
mean_val = df['mqtt.msgtype'].mean()
median_val = df['mqtt.msgtype'].median()
mode_val = df['mqtt.msgtype'].mode()[0]
skewness = skew(df['mqtt.msgtype'].dropna())

print(f"Central Tendency Measures for MQTT Message Types:")
print(f"Mean: {mean_val:.2f}")
print(f"Median: {median_val:.2f}")
print(f"Mode: {mode_val}")
print(f"Skewness: {skewness:.2f}")

msg_type_map = {
    1: "CONNECT", 2: "CONNACK", 3: "PUBLISH", 4: "PUBACK",
    5: "PUBREC", 6: "PUBREL", 7: "PUBCOMP", 8: "SUBSCRIBE",
    9: "SUBACK", 10: "UNSUBSCRIBE", 11: "UNSUBACK",
    12: "PINGREQ", 13: "PINGRESP", 14: "DISCONNECT"
}


if 'target' in df.columns:
    plt.figure(figsize=(14, 8))
    attack_types = df['target'].unique()
    x_pos = np.arange(len(attack_types))
    width = 0.35
    mean_by_attack = df.groupby('target')['mqtt.msgtype'].mean()
    bars = plt.bar(x_pos, mean_by_attack, width, label='Mean Message Type', color='skyblue')

    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{mean_by_attack.iloc[i]:.2f}", ha='center', fontweight='bold')
    

    skew_by_attack = df.groupby('target')['mqtt.msgtype'].apply(lambda x: skew(x.dropna()))
    for i, attack in enumerate(attack_types):
        skew_val = skew_by_attack[attack]
        if skew_val > 0.5:
            skew_text = f"Skew: {skew_val:.2f} (Right)"
            color = 'darkred'
        elif skew_val < -0.5:
            skew_text = f"Skew: {skew_val:.2f} (Left)"
            color = 'darkblue'
        else:
            skew_text = f"Skew: {skew_val:.2f} (Sym)"
            color = 'darkgreen'
        
        plt.text(i, 0.2, skew_text, ha='center', color=color, fontweight='bold')
    
    plt.axhline(y=mean_val, color='red', linestyle='--', 
                label=f'Overall Mean: {mean_val:.2f}')

    plt.xlabel('Attack Category', fontsize=12)
    plt.ylabel('Mean Message Type', fontsize=12)
    plt.title('Mean MQTT Message Types by Attack Category', fontsize=14)
    plt.xticks(x_pos, attack_types)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')

    ax2 = plt.twinx()
    max_msg_type = int(np.ceil(mean_by_attack.max())) + 1
    ax2.set_ylim(0, max_msg_type)
    ax2.set_yticks(range(1, max_msg_type + 1))
    ax2.set_yticklabels([f"{i}: {msg_type_map.get(i, '')}" for i in range(1, max_msg_type + 1)])
    ax2.set_ylabel('MQTT Message Type', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('mqtt_attack_mean_types.png')
    plt.show()