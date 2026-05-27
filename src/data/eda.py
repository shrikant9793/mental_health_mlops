import pandas as pd
import yaml
import matplotlib.pyplot as plt

# Load config
with open(r"configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load data
df = pd.read_csv(config["data"]["raw_data_path"])

# Basic Info
print("=== Dataset Info ===")
print(f"Total Rows    : {len(df)}")
print(f"Total Columns : {len(df.columns)}")
print(f"Columns       : {df.columns.tolist()}")
print(f"Null Values   :\n{df.isnull().sum()}")

# Class Distribution
print("\n=== Class Distribution ===")
print(df[config["model"]["target_column"]].value_counts())

# Text Length Analysis
df["text_length"] = df[config["model"]["text_column"]].apply(len)
print("\n=== Text Length Stats ===")
print(df["text_length"].describe())

# Plot 1 — Class Distribution
plt.figure(figsize=(6, 4))
df[config["model"]["target_column"]].value_counts().plot(kind="bar", color=["steelblue", "salmon"])

plt.title("Class Distribution")
plt.xlabel("is_depression")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("reports/class_distribution.png")
print("\n✅ Class distribution plot saved to reports/")

# Plot 2 — Text Length Distribution
plt.figure(figsize=(6, 4))
df["text_length"].hist(bins=50, color="steelblue")
plt.title("Text Length Distribution")
plt.xlabel("Text Length")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("reports/text_length_distribution.png")
print("✅ Text length distribution plot saved to reports/")