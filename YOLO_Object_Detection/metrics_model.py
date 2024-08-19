import pandas as pd
import matplotlib.pyplot as plt

# Path to the CSV file containing training metrics
csv_file_path = '/content/runs/detect/train/results.csv'

# Load CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Strip leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Create a single figure with subplots
plt.figure(figsize=(10, 8))  # Adjust figure size to be more compact

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Plot training and validation losses
plt.subplot(2, 2, 1)  # 2 rows, 2 columns, first subplot
plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
plt.plot(df['epoch'], df['val/box_loss'], label='Validation Box Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Box Loss')
plt.legend(fontsize='small')

plt.subplot(2, 2, 2)  # 2 rows, 2 columns, second subplot
plt.plot(df['epoch'], df['train/cls_loss'], label='Train Classification Loss')
plt.plot(df['epoch'], df['val/cls_loss'], label='Validation Classification Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Classification Loss')
plt.legend(fontsize='small')

plt.subplot(2, 2, 3)  # 2 rows, 2 columns, third subplot
plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.title('Precision and Recall')
plt.legend(fontsize='small')

# Plot mAP metrics in the same figure
plt.subplot(2, 2, 4)  # 2 rows, 2 columns, fourth subplot
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@50')
plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@50-95')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.title('mAP Metrics')
plt.legend(fontsize='small')

# Save the figure to a specific folder
output_path = '/content'
plt.savefig(output_path, dpi=300, bbox_inches='tight')

# Display all plots in one window
plt.show()
