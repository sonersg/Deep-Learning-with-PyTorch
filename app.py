# import torch
# print(torch.cuda.is_available())  # should return True


# print(torch.__version__)  # should return the installed PyTorch version

# mt = torch.arange(10)
# # print(mt.reshape(2, 5))
# mt.reshape(2, 5)

# tensor_a = torch.tensor([1,2,3,4])
# tensor_b = torch.tensor([2,3,4,5])

# # print(tensor_a + tensor_b)
# result = torch.add(tensor_a, tensor_b)
# print(result)

import torch
import torch.nn as nn
import torch.nn.functional as F


# Create a Model class thta inherits nn.Module
class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


# Pick a manual seed for randomization
torch.manual_seed(4)

model = Model()

import pandas as pd

pd.set_option("future.no_silent_downcasting", True)
import matplotlib.pyplot as plt

# %matplotlib inline

url = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
my_df = pd.read_csv(url)

# print(my_df)

# Change last column from strings to integers
# In your preprocessing (around line 53)
my_df["variety"] = my_df["variety"].replace(
    {"Setosa": 0, "Versicolor": 1, "Virginica": 2}
)

# print(my_df)

# Train, Test, Split! Set x, y
X = my_df.drop("variety", axis=1)
y = my_df["variety"]

# Convert these to numpy arrays
X = X.values
y = y.values
# print(X)

# Train, Test, Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=41
)

# Convert X features to float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
# Convert y labels to tensors long
y_train = torch.LongTensor(y_train.astype(int))
y_test = torch.LongTensor(y_test.astype(int))

# Set the criterion of model to measure the error, how far r the predictions off from the data
criterion = nn.CrossEntropyLoss()
# Choose Adam Optimizer, lr = learning rate (if error doesn't go down after bunch of iterations)
# Choose Adam Optimizer, lr = learning rate (epochs), lower our learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# print(model.parameters)

# Train our model
# Epochs?: One run through all the training data in our network
epochs = 222
losses = []
for i in range(epochs):
    # Go forward and get a prediction
    y_pred = model.forward(X_train)  # Get predicted result

    # Measure the loss, error (gonna be high at first)
    loss = criterion(y_pred, y_train)  # Predixted values vs y_train

    # Kepp track of our losses
    losses.append(loss.detach().numpy())

    # Print every 10 epoch
    if i % 10 == 0:
        print(f"Epoch: {i} and loss: {loss}")

    # Do some back propogation: take the error rate of forward propogation
    # and feed it back through the network to fine tune the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Graph it out
# plt.plot(range(epochs), losses)
# plt.ylabel("loss/error")
# plt.xlabel("Epoch")
# plt.show()

# Evaluate model on test data set
with torch.no_grad():  # Basically turn off back propogation
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test)  # Find loss, error

# print(loss)

correct = 0
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)
        if y_test[i] == 0:
            x = "Setosa"
        elif y_test[i] == 1:
            x = "Versicolor"
        else:
            x = "Virginica"
        # Will tell us what type of flower our network thinks it is
        # print(f'{i+1}, {str(y_val)} \t {y_test[i]} \t {y_val.argmax().item()}')
        print(f"{i+1}, {str(y_val)} \t {x}")
        # Correct or not
        if y_val.argmax().item() == y_test[i]:
            correct += 1

print(f"We have {correct} correct!")

new_iris = torch.tensor([4.7, 3.2, 1.3, 0.2])
with torch.no_grad():
    print(model(new_iris))

new_iris2 = torch.tensor([5.9, 3.0, 5.1, 1.8])
with torch.no_grad():
    print(model(new_iris2))

# Save our NN Model
torch.save(model.state_dict(), "my_iris.pt")
# Load the saved Model
saved_model = Model()
saved_model.load_state_dict(torch.load("my_iris.pt"))
# Make sure it loaded correctly
saved_model.eval()
