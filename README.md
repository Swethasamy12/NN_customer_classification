# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Write your own steps

### STEP 2:

### STEP 3:


## PROGRAM

### Name: SWETHA C
### Register Number: 212224230283

```
class PeopleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
```
# Initialize the Model, Loss Function, and Optimizer
input_size = X_train.shape[1]
num_classes = 4

model = PeopleClassifier(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```
```
def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
```


## Dataset Information

<img width="1149" height="465" alt="image" src="https://github.com/user-attachments/assets/9c8d999d-e182-4ab5-b78b-6fe03afbc4a4" />


## OUTPUT



### Confusion Matrix
<img width="699" height="611" alt="image" src="https://github.com/user-attachments/assets/c0c7b721-ca94-4f93-a989-f5e0c506988a" />


### Classification Report

<img width="571" height="241" alt="image" src="https://github.com/user-attachments/assets/f4b9ff67-4846-4109-ab1c-a6676fa23a72" />



### New Sample Data Prediction

<img width="779" height="258" alt="image" src="https://github.com/user-attachments/assets/791b4614-e715-4c74-919f-2e2651efe087" />


## RESULT
Thus the neural network classification model was successfully developed.
