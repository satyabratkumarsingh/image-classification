import tensorflow as tf
import keras
from tensorflow.keras.layers import Input, Flatten
from  utils.image_processor import run_dlib_shape, extract_features_labels
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy

class ANN(nn.Module):

    def __init__(self):
        super().__init__()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(68*2, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flat(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

def get_data():
    X, y = extract_features_labels()
  
    Y = np.array([y, -(y - 1)]).T
    tr_X = X[:100] ; tr_Y = Y[:100]
    te_X = X[100:] ; te_Y = Y[100:]

    return tr_X, tr_Y, te_X, te_Y


def train_pytorch():
    learning_rate = 0.00001
    training_epochs = 500
    accuracy_metric = Accuracy(task="binary")

    training_images, training_labels, test_images, test_labels = get_data()
    training_labels = np.argmax(training_labels, axis=1)
    test_labels = np.argmax(test_labels, axis=1)

    training_images = torch.tensor(training_images, dtype=torch.float32)
    training_labels = torch.tensor(training_labels, dtype=torch.float32)
    test_images = torch.tensor(test_images, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)

    print(f'Shape of training_images : {training_images.shape}')
    print(f'Shape of training_labels : {training_labels.shape}')
    print(f'Shape of test_images : {test_images.shape}')
    print(f'Shape of test_labels : {test_labels.shape}')

    model = ANN()
    print(model)
    loss_fn   = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(training_epochs):
        model.train()
        optimizer.zero_grad()
      
        outputs = model(training_images)
        loss = loss_fn(outputs, training_labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()

        if epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(test_images)
                test_loss = loss_fn(test_outputs, test_labels.unsqueeze(1).float())
                train_acc = accuracy_metric(outputs, training_labels.unsqueeze(1).float())
                test_acc = accuracy_metric(test_outputs, test_labels.unsqueeze(1).float())

            print(f'Epoch {epoch+1}/{training_epochs}, '
                  f'Training Loss: {loss.item():.4f}, '
                  f'Test Loss: {test_loss.item():.4f}, '
                  f'Training Accuracy: {train_acc.item():.4f}, '
                  f'Test Accuracy: {test_acc.item():.4f}')
