from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)


in_features, out_features = X_train.shape

inputs = torch.tensor(X_train, dtype=torch.float32)

labels = torch.tensor(y_train, dtype=torch.float32)
linear = nn.Linear(30, 1)

optimizer = optim.SGD(linear.parameters(), lr = 0.1)

criterion = nn.MSELoss()

epochs = 5000

for epoch in range(epochs):
    optimizer.zero_grad()


    pred = torch.sigmoid(linear(inputs))

    loss = criterion(pred, labels)

    loss.backward()

    optimizer.step()


X_test_tensor = torch.tensor(X_test, dtype = torch.float32)

pred = linear(X_test_tensor)

labels_predicted = (pred >= 0.5).float()

corrects = 0
wrongs = 0

for k in range(len(labels_predicted)):
    print(f"Predicted {int(labels_predicted[k].item())} - Real {y_test[k]}")
    if (int(labels_predicted[k].item()) == y_test[k]):
        corrects += 1 
    else:
        wrongs += 1

print(f"Corrects - {corrects}")
print(f"Wrongs - {wrongs}")

print(f"Accuracy - {round(corrects/(wrongs + corrects) * 100, 1)}%")

