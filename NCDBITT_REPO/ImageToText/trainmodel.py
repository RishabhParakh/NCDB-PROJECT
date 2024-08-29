import os
import re
import ssl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from torchvision.models import ResNet18_Weights
from datetime import datetime

# Get the current time
current_time = datetime.now().time()

# Print the current time
print("Current Time:", current_time)

# Disable SSL certificate verification globally
ssl._create_default_https_context = ssl._create_unverified_context

# Define the car classes
car_classes = ["Acura_ILX", "Acura_MDX", "Alfa Romeo", "Aston Martin_DB", "Audi_A",
               "Bentley_Continental GT", "BMW", "Cadillac_ATS", "Cadillac_CT", 
               "Cadillac_CTS", "Cadillac_Escalade", "Cadillac_XT", "Cadillac_XTS"]

# Define the Dataset Class
class CarDataset(Dataset):
    def __init__(self, image_dir, transform=None, limit=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        if limit:
            self.image_files = self.image_files[:limit]
        self.labels = [self.extract_model_name(f) for f in self.image_files]
        self.label_to_idx = {label: idx for idx, label in enumerate(car_classes)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def extract_model_name(self, filename):
        # Extract the car model name by matching against known car classes
        for car_class in car_classes:
            if car_class in filename:
                return car_class
        return "Unknown"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        label = self.label_to_idx.get(label, len(car_classes))  # "Unknown" is the last class

        if self.transform:
            image = self.transform(image)

        return image, label

# Define the CNN Model Class
class CarModelCNN(nn.Module):
    def __init__(self, num_classes=len(car_classes) + 1):  # Including "Unknown" class
        super(CarModelCNN, self).__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Training Function
def train_model(model, dataloader, criterion, optimizer, num_epochs=75):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss * 100:.2f}%')

# Function to calculate accuracy on the entire dataset
def calculate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on the entire dataset: {accuracy:.2f}%')

# Main Script
if __name__ == "__main__":
    # Parameters
    image_dir = '/home/ncdbproj/NCDBContent/ImageToText/car_dataset2/60000_carimages/60000_carimages/carcar'
    batch_size = 16
    num_epochs = 10
    learning_rate = 0.0001

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Data
    dataset = CarDataset(image_dir=image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize Model, Loss, and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CarModelCNN(num_classes=len(car_classes) + 1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the Model
    train_model(model, dataloader, criterion, optimizer, num_epochs=num_epochs)

    # Save the Model
    torch.save(model.state_dict(), '/home/ncdbproj/NCDBContent/ImageToText/car_model_recognition.pth')

    # Calculate accuracy on the entire dataset
    calculate_accuracy(model, dataloader)

