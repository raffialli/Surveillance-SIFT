import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from model_definition import NN


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 128 # Changed from 64 to increase training speed
learning_rate = 1e-3
num_epochs = 15
num_classes = 2
input_size = 3 * 320 * 180  # Updated to match your image size

# Data paths
train_path = "E:\\TensorFlow\\Front Door Vids\\training-files\\training"
val_path = "E:\\TensorFlow\\Front Door Vids\\training-files\\validation"


transform = transforms.Compose([
    transforms.Resize((320, 180)),  # Resize the image to the expected input size
    transforms.ToTensor()  # Convert the PIL Image to a tensor
])

train_dataset = ImageFolder(root=train_path, transform=transform)
val_dataset = ImageFolder(root=val_path, transform=transform)



# Handle class imbalance
class_count = [5534, 15284]
n_samples = sum(class_count)
class_weights = [n_samples / count for count in class_count]
samples_weights = [class_weights[label] for _, label in train_dataset]
sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

# Create DataLoader with the sampler
train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = NN(input_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Early stopping details
n_epochs_stop = 5
min_val_loss = float('inf')
epochs_no_improve = 0

# Record the start time
start_time = time.time()

# Train Network
for epoch in range(num_epochs):
    losses = []
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        # Flatten the data
        data = data.reshape(data.shape[0], -1)
        
        # forward
        scores = model(data)
        loss = criterion(scores, targets)
        
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    # Check validation loss at end of each epoch
    val_loss = 0
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data, target in val_loader:
            data = data.to(device=device)
            target = target.to(device=device)
            data = data.reshape(data.shape[0], -1)
            output = model(data)
            val_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    model.train()

    # Average validation loss
    val_loss /= len(val_loader)

    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    print(f'Validation Accuracy: {(100 * correct / total):.2f}%')
    
    # Record the end time
    end_time = time.time()
    
    # Early stopping
    if val_loss < min_val_loss:
        epochs_no_improve = 0
        min_val_loss = val_loss
        
        # Save the model
        torch.save(model, 'FrontDoor_new_dataset_v4.pth')
        
    else:
        epochs_no_improve += 1
        if epochs_no_improve == n_epochs_stop:
            print('Early stopping!')
            break
# Calculate and print total runtime
total_time = end_time - start_time
print("-------------------------------")
print(f'Total runtime: {total_time / 60:.2f} minutes')
