import matplotlib.pyplot as plt
import pandas as pd

# Data
df = pd.read_csv("RandomForest.csv")
print(df.columns)
# Create bar plot
plt.bar(df["Feature"], df[" Importance"], color='skyblue', edgecolor='black')

# Customize
plt.title('Basic Bar Plot')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=90)  # Rotate labels vertically
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()  # Prevents label cutoff
plt.show()
