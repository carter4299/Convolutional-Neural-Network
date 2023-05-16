import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn

#-----------------------------------------------------------
# Image Transformation
#-----------------------------------------------------------

transform_pipeline = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

#-----------------------------------------------------------
#Prepare train and test set by loading CIFAR10 dataset from torchvision.datasets.
#-----------------------------------------------------------

from torchvision import datasets
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_pipeline)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_pipeline)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

#-----------------------------------------------------------
# Use torch.utils.data.random_split() to make a validation set from the training set with 80:20 split
#-----------------------------------------------------------

from torch.utils.data import random_split

train_size = int(0.8 * len(train_dataset))
valid_size = len(train_dataset) - train_size

train_set, valid_set = random_split(train_dataset, [train_size, valid_size])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True, num_workers=2)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=100, shuffle=False, num_workers=2)

#-----------------------------------------------------------
# Prepare three dataloaders for train, validation and test set. Use an appropriate batchsize of your choice
#-----------------------------------------------------------

from os import truncate

#set a batch size
batch_size = 128

#write dataloader for train set
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=truncate, num_workers=2)

#write dataloader for test set
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

#write dataloader for validation set
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2)

#-----------------------------------------------------------
#Load a random batch of images from the training set using the trainloader. Then use make_grid() from torchvision.utils and imshow() from matplotlib.pyplot to show the images. Also, print the corresponding true labels for those image samples
#-----------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

#load a random batch of test set images
dataiter = iter(train_loader)
images, labels = next(dataiter)
grid = make_grid(images)
grid = grid.numpy().transpose((1, 2, 0))

#show the images
plt.imshow(grid)
plt.show()

#print the ground truth class labels for these images
print('ground truth class labels:', labels)

#-----------------------------------------------------------
#Define a neural network model
#-----------------------------------------------------------

import torch.nn.functional as F

class Carter(nn.Module):
    def __init__(self):
        #Initialize the layers
        super(Carter, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, 10)

    def forward(self, x):
        #Define the dataflow through the layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

#-----------------------------------------------------------
#Create an instance of the model class that you just prepared
#-----------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Carter().to(device)

#-----------------------------------------------------------
#Set up Cross Entropy Loss as the loss function and Adam as the optimizer. Use a learning rate of your choice for the optimizer
#-----------------------------------------------------------

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#-----------------------------------------------------------
#Write a training loop to load data, compute model output, compute loss and backpropagating it to update model parameters
#-----------------------------------------------------------

#Define number of epochs
num_epochs = 50

#Initialize empty lists to store training loss, training accuracy, validation loss, validation accuracy
# You will use these lists to plot the loss history.
train_losses = []
train_accuracies = []
valid_losses = []
valid_accuracies = []

#Loop through the number of epochs
for epoch in range(num_epochs):
    #set model to train mode
    model.train()

    #iterate over the training data in batches
    running_train_loss = 0.0
    running_train_corrects = 0
    #get the image inputs and labels from current batch
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        #set the optimizer gradients to zero to avoid accumulation of gradients
        optimizer.zero_grad()

        #compute the output of the model
        outputs = model(inputs)

        #compute the loss on current batch
        loss = criterion(outputs, labels)

        #backpropagate the loss
        loss.backward()

        #update the optimizer parameters
        optimizer.step()

        #update the train loss and accuracy
        running_train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        running_train_corrects += torch.sum(preds == labels.data)

    #compute the average training loss and accuracy and store in respective arrays
    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    epoch_train_acc = running_train_corrects.double() / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)

    #set the model to evaluation mode
    model.eval()

    #keeping the gradient computation turned off, loop over batches of data from validation set.
    running_valid_loss = 0.0
    running_valid_corrects = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            #compute output of the model
            outputs = model(inputs)

            #compute the loss
            loss = criterion(outputs, labels)

            #compute the validation loss and accuracy
            running_valid_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_valid_corrects += torch.sum(preds == labels.data)

    #compute the average validation loss and accuracy and store in respective lists
    epoch_valid_loss = running_valid_loss / len(valid_loader.dataset)
    epoch_valid_acc = running_valid_corrects.double() / len(valid_loader.dataset)
    valid_losses.append(epoch_valid_loss)
    valid_accuracies.append(epoch_valid_acc)

    #print the training loss, training accuracy, validation loss and validation accuracy at the end of each epoch
    print(
        f'Epoch {epoch + 1}/{num_epochs}: train_loss={epoch_train_loss:.4f}, train_acc={epoch_train_acc:.4f}, valid_loss={epoch_valid_loss:.4f}, valid_acc={epoch_valid_acc:.4f}')

    #save the model parameters once in every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')

#-----------------------------------------------------------
#Plot and compare
#-----------------------------------------------------------

#plot the training and validation loss
plt.plot(train_losses, label='training Loss')
plt.plot(valid_losses, label='validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('training vs. validation Loss')
plt.legend()
plt.show()


#plot the training and validation accuracy
plt.plot(train_accuracies, label='training Accuracy')
plt.plot(valid_accuracies, label='validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('training vs. validation Accuracy')
plt.legend()
plt.show()

#-----------------------------------------------------------
#Load the best performing model (one with good validation accuracy and without overfitting) among the ones you saved
#-----------------------------------------------------------

#instantiate a model
best_model = Carter()

#load parameters from one of the saved model states
best_model.load_state_dict(torch.load(f'model_epoch_35.pth'))

#set this model to evaluation mode
best_model.eval()

#-----------------------------------------------------------
#Take a random batch of images from test set and show the images. Print the corresponding ground truth class labels. Then compute model output (model selected at previous step) and the predicted labels for the images in this batch
#-----------------------------------------------------------

import random

#load a random batch of test set images
random_batch = random.randint(0, len(test_loader) - 1)
dataiter = iter(test_loader)
for i in range(random_batch):
    images, labels = next(dataiter)

#show the images
grid = make_grid(images)
grid = grid.numpy().transpose((1, 2, 0))
plt.imshow(grid)
plt.show()

#print the ground truth class labels for these images
class_names = test_dataset.classes
ground_truth_labels = [class_names[label] for label in labels]
print('Ground truth class labels:', ground_truth_labels)

#compute model output
images = images.to(device)
outputs = best_model(images)

#print the predicted class labels for these images
_, predicted_labels = torch.max(outputs, 1)
predicted_labels = [class_names[label] for label in predicted_labels]
print('Predicted class labels:', predicted_labels)

#-----------------------------------------------------------
#Compute the average accuracy on test data using this model
#-----------------------------------------------------------

#compute accuracy on each batch of test set
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = best_model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

#print the average accuracy
average_accuracy = 100 * correct / total
print(f'Average accuracy: {average_accuracy:.2f}%')

#-----------------------------------------------------------
# Compute the average accuracy for each individual class
#-----------------------------------------------------------

#compute per-class accuracy on each batch of test set
per_class_correct = {class_name: 0 for class_name in class_names}
per_class_total = {class_name: 0 for class_name in class_names}

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = best_model(images)
        _, predicted = torch.max(outputs, 1)

        for i in range(labels.size(0)):
            label = labels[i]
            class_name = class_names[label]
            per_class_correct[class_name] += (predicted[i] == label).item()
            per_class_total[class_name] += 1

#print per-class accuracy for 10 output classes
print("Per-class accuracy:")
for class_name in class_names:
    accuracy = 100 * per_class_correct[class_name] / per_class_total[class_name]
    print(f'{class_name}: {accuracy:.2f}%')

#-----------------------------------------------------------