# import torch
# import torch.nn as nn
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Data loading and transformation
# data_transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((200, 200)),
#     transforms.ToTensor()]) 

# test_dataset = datasets.ImageFolder(root='/home/hankla/Desktop/work/puddle_deploy/data_train/data_24_10_now/test', transform=data_transform)

# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)  # No need to shuffle for evaluation

# # CNN Model definition
# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes): #classi_puddle
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(32 * 25 * 25, 16)
#         self.relu = nn.ReLU()  
#         self.fc2 = nn.Linear(16, num_classes)

#     def forward(self, x):
#         x = self.pool1(self.relu(self.conv1(x)))
#         x = self.pool2(self.relu(self.conv2(x)))
#         x = self.pool3(self.relu(self.conv3(x)))
#         x = x.view(-1, 32 * 25 * 25)
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))  
#         return x

# # Initialize the model
# model = SimpleCNN(num_classes=2)
# model.load_state_dict(torch.load('/home/hankla/Desktop/work/puddle_deploy/yolov8_models/classi_puddle.pth', map_location=torch.device('cpu')))  # Load trained weights onto CPU

# # Evaluate the model
# model.eval()
# all_labels = []
# all_predictions = []

# with torch.no_grad():
#     for inputs, labels in test_loader:
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         all_labels.extend(labels.numpy())
#         all_predictions.extend(predicted.numpy())

# # Generate confusion matrix
# cm = confusion_matrix(all_labels, all_predictions)

# # Plot confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
# plt.xlabel("Predicted Labels")
# plt.ylabel("True Labels")
# plt.title("Confusion Matrix")
# plt.show()

# # Calculate precision, recall, and F1-score for each class
# class_precision = precision_score(all_labels, all_predictions, average=None)
# class_recall = recall_score(all_labels, all_predictions, average=None)
# class_f1 = f1_score(all_labels, all_predictions, average=None)

# # Print precision, recall, and F1-score for each class
# for i, (precision, recall, f1) in enumerate(zip(class_precision, class_recall, class_f1)):
#     print(f"Class {i}: Precision={precision:.4f}, Recall={recall:.4f}, F1-score={f1:.4f}")

# # Calculate macro F1
# macro_f1 = f1_score(all_labels, all_predictions, average='macro')
# print(f"Macro F1-score: {macro_f1:.8f}")

# # Calculate micro F1
# micro_f1 = f1_score(all_labels, all_predictions, average='micro')
# print(f"Micro F1-score: {micro_f1:.8f}")

# # Prepare data for table
# num_classes = len(class_precision)
# class_labels = [f'Class {i}' for i in range(num_classes)]
# metrics = ['Precision', 'Recall', 'F1-score']
# class_metrics = list(zip(class_precision, class_recall, class_f1))

# # Add macro F1-score and micro F1-score
# class_labels.extend(['Macro Avg', 'Micro Avg'])
# class_metrics.extend([(macro_f1, macro_f1, macro_f1), (micro_f1, micro_f1, micro_f1)])

# # Create table data
# table_data = [[label] + list(metrics) for label, metrics in zip(class_labels, class_metrics)]

# # Create table
# plt.figure(figsize=(10, 6))
# table = plt.table(cellText=table_data, colLabels=['Class'] + metrics, loc='center')

# # Adjust table font size
# table.auto_set_font_size(False)
# table.set_fontsize(10)

# # Hide axes
# ax = plt.gca()
# ax.axis('off')


# plt.figure(figsize=(10, 6))
# plt.plot(range(num_classes), class_precision, marker='o', label='Precision')
# plt.plot(range(num_classes), class_recall, marker='o', label='Recall')
# plt.plot(range(num_classes), class_f1, marker='o', label='F1-score')
# plt.xlabel('Class')
# plt.ylabel('Score')
# plt.title('Metrics Comparison for Each Class')
# plt.xticks(range(num_classes), class_labels[:-2] + ['Macro Avg', 'Micro Avg'], rotation=45)
# plt.legend()
# plt.grid(True)
# plt.show()




# # Show plot
# plt.title('Metrics Table')
# plt.show()


import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Data loading and transformation
data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((200, 200)),
    transforms.ToTensor()]) 

test_dataset = datasets.ImageFolder(root='/home/hankla/Desktop/work/puddle_deploy/data_train/data_24_10_now/test', transform=data_transform)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)  # No need to shuffle for evaluation

# CNN Model definition
class SimpleCNN(nn.Module):
    def __init__(self, num_classes): #classi_puddle
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 25 * 25, 16)
        self.relu = nn.ReLU()  
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = self.pool3(self.relu(self.conv3(x)))
        x = x.view(-1, 32 * 25 * 25)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))  
        return x

# Initialize the model
model = SimpleCNN(num_classes=2)
model.load_state_dict(torch.load('/home/hankla/Desktop/work/puddle_deploy/yolov8_models/classi_puddle.pth', map_location=torch.device('cpu')))  # Load trained weights onto CPU

# Evaluate the model
model.eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.numpy())
        all_predictions.extend(predicted.numpy())

# Generate confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Calculate precision, recall, and F1-score for each class
class_precision = precision_score(all_labels, all_predictions, average=None)
class_recall = recall_score(all_labels, all_predictions, average=None)
class_f1 = f1_score(all_labels, all_predictions, average=None)

# Print precision, recall, and F1-score for each class
for i, (precision, recall, f1) in enumerate(zip(class_precision, class_recall, class_f1)):
    print(f"Class {i}: Precision={precision:.4f}, Recall={recall:.4f}, F1-score={f1:.4f}")

# Calculate macro F1
macro_f1 = f1_score(all_labels, all_predictions, average='macro')
print(f"Macro F1-score: {macro_f1:.8f}")

# Calculate micro F1
micro_f1 = f1_score(all_labels, all_predictions, average='micro')
print(f"Micro F1-score: {micro_f1:.8f}")

# Prepare data for table
num_classes = len(class_precision)
class_labels = [f'Class {i}' for i in range(num_classes)]
metrics = ['Precision', 'Recall', 'F1-score']
class_metrics = list(zip(class_precision, class_recall, class_f1))

# Add macro F1-score and micro F1-score
class_labels.extend(['Macro Avg', 'Micro Avg'])
class_metrics.extend([(macro_f1, macro_f1, macro_f1), (micro_f1, micro_f1, micro_f1)])

# Create table data
table_data = [[label] + list(metrics) for label, metrics in zip(class_labels, class_metrics)]

# Create table
plt.figure(figsize=(10, 6))
table = plt.table(cellText=table_data, colLabels=['Class'] + metrics, loc='center')

# Adjust table font size
table.auto_set_font_size(False)
table.set_fontsize(10)

# Hide axes
ax = plt.gca()
ax.axis('off')

# Show plot
plt.title('Metrics Table')
plt.show()


import numpy as np

# Define classes
classes = ['None Puddle', 'Puddle']

# Define metrics
metrics = ['Precision', 'Recall', 'F1-score']

# Define values for class 0 and class 1
class_0_values = [precision_score(all_labels, all_predictions, average=None)[0],
                  recall_score(all_labels, all_predictions, average=None)[0],
                  f1_score(all_labels, all_predictions, average=None)[0]]

class_1_values = [precision_score(all_labels, all_predictions, average=None)[1],
                  recall_score(all_labels, all_predictions, average=None)[1],
                  f1_score(all_labels, all_predictions, average=None)[1]]

# Plot bar chart
x = np.arange(len(metrics))
width = 0.3

fig, ax = plt.subplots(figsize=(10, 6))
bars_class_0 = ax.bar(x - width/2, class_0_values, width, label=classes[0])
bars_class_1 = ax.bar(x + width/2, class_1_values, width, label=classes[1])

# Add labels, title, and legend
ax.set_ylabel('Scores')
ax.set_title('Comparison of Metrics for Class None puddle and Class Puddle')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Add text annotations
def add_annotations(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}', 
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # ปรับตำแหน่งข้อความให้อยู่ตรงกลางบาร์
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10)  # Adjust fontsize here

add_annotations(bars_class_0)
add_annotations(bars_class_1)

# Add numerical labels at the center of each bar
for bar1, bar2 in zip(bars_class_0, bars_class_1):
    ax.text(bar1.get_x() + bar1.get_width() / 2, bar1.get_height() / 2, f'{bar1.get_height():.4f}', 
            ha='center', va='center', color='black', fontsize=20)
    ax.text(bar2.get_x() + bar2.get_width() / 2, bar2.get_height() / 2, f'{bar2.get_height():.4f}', 
            ha='center', va='center', color='black', fontsize=10)

plt.show()
