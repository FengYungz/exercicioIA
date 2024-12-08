import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchinfo import summary

# Step 1: Load the dataset
file_path = "in-vehicle-coupon-recommendation.csv"  # Ensure this file is in the same directory as the script
data = pd.read_csv(file_path)

# Step 2: Explore the dataset
print("Dataset head:")
print(data.head())
print("\nDataset Info:")
data.info()

# Step 3: Preprocessing
# Handle missing values
data = data.dropna()

# Encode categorical features
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Normalize numerical features
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Define features and target
X = data.drop(columns=['Y']).values  # Assuming 'Y' is the target column
y = data['Y'].values.astype(int)  # Ensure y is binary and categorical

# Step 4: Define a function for the StratifiedKFold split and training
def stratified_kfold_train_evaluate(model_class, model_name, X, y, epochs=20, batch_size=32, lr=0.001):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    loss_per_epoch = []
    fold = 1

    for train_index, val_index in skf.split(X, y):
        print(f"Training fold {fold}...")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)

        # Define model, loss, and optimizer
        model = model_class(X_train_tensor.shape[1])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training loop
        epoch_losses = []
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        loss_per_epoch.append(epoch_losses)

        # Evaluation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            _, val_preds = torch.max(val_outputs, 1)
            accuracy = accuracy_score(y_val_tensor, val_preds)
            precision = precision_score(y_val_tensor, val_preds, average='binary')
            recall = recall_score(y_val_tensor, val_preds, average='binary')

        fold_metrics.append((accuracy, precision, recall))
        fold += 1

    # Average metrics across folds
    avg_accuracy = np.mean([m[0] for m in fold_metrics])
    avg_precision = np.mean([m[1] for m in fold_metrics])
    avg_recall = np.mean([m[2] for m in fold_metrics])

    print(f"{model_name} Average Metrics:\nAccuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}")

    # Plot loss curve
    plt.figure()
    for fold_idx, fold_loss in enumerate(loss_per_epoch, 1):
        plt.plot(fold_loss, label=f"Fold {fold_idx}")
    plt.title(f"Loss Curve for {model_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    return avg_accuracy, avg_precision, avg_recall

# Step 5: Define models
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.linear(x)

class MLPModel(nn.Module):
    def __init__(self, input_dim):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return self.fc3(x)

class DeepLearningModel(nn.Module):
    def __init__(self, input_dim):
        super(DeepLearningModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu(x)
        return self.fc4(x)

# Visualize model architectures
def visualize_model(model_class, input_dim):
    model = model_class(input_dim)
    print(summary(model, input_size=(1, input_dim)))

# Step 6: Train and evaluate models
print("\nEvaluating Logistic Regression Model...")
visualize_model(LogisticRegressionModel, X.shape[1])
stratified_kfold_train_evaluate(LogisticRegressionModel, "Logistic Regression", X, y)

print("\nEvaluating MLP Model...")
visualize_model(MLPModel, X.shape[1])
stratified_kfold_train_evaluate(MLPModel, "MLP", X, y)

print("\nEvaluating Deep Learning Model...")
visualize_model(DeepLearningModel, X.shape[1])
stratified_kfold_train_evaluate(DeepLearningModel, "Deep Learning", X, y)
