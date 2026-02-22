import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pygame
import numpy as np
import cv2

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Load MNIST dataset

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

import os

model = CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model_path = 'cnn_mnist.pth'

def train_model(epochs):
    """Train the model for the specified number of epochs"""
    print(f"Training for {epochs} epoch(s)...")
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, loss: {running_loss/len(trainloader):.4f}")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")
    test_model()

def load_trained_model():
    """Load the pre-trained model"""
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"Loaded trained model from {model_path}")
        test_model()
        return True
    else:
        print(f"No trained model found at {model_path}")
        return False

def test_model():
    """Test the model accuracy"""
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    print(f"Accuracy: {100*correct/total:.2f}%")

def show_menu():
    """Show menu to choose training option"""
    pygame.init()
    screen = pygame.display.set_mode((500, 400))
    pygame.display.set_caption("Choose Training Option")
    clock = pygame.time.Clock()
    
    font = pygame.font.Font(None, 32)
    title_font = pygame.font.Font(None, 40)
    
    # Define buttons
    button_width = 400
    button_height = 60
    button_x = 50
    
    button1_rect = pygame.Rect(button_x, 80, button_width, button_height)
    button2_rect = pygame.Rect(button_x, 160, button_width, button_height)
    button3_rect = pygame.Rect(button_x, 240, button_width, button_height)
    
    while True:
        screen.fill((30, 30, 30))
        
        # Draw title
        title = title_font.render("Select Training Mode", True, (255, 255, 255))
        screen.blit(title, (80, 20))
        
        # Draw buttons
        mouse_pos = pygame.mouse.get_pos()
        
        # Button 1: Use trained model
        color1 = (70, 130, 180) if button1_rect.collidepoint(mouse_pos) else (50, 100, 150)
        pygame.draw.rect(screen, color1, button1_rect, border_radius=10)
        text1 = font.render("Use Trained Model", True, (255, 255, 255))
        screen.blit(text1, (button_x + 85, 100))
        
        # Button 2: Train 1 epoch
        color2 = (70, 130, 180) if button2_rect.collidepoint(mouse_pos) else (50, 100, 150)
        pygame.draw.rect(screen, color2, button2_rect, border_radius=10)
        text2 = font.render("Train 1 Epoch", True, (255, 255, 255))
        screen.blit(text2, (button_x + 110, 180))
        
        # Button 3: Train 10 epochs
        color3 = (70, 130, 180) if button3_rect.collidepoint(mouse_pos) else (50, 100, 150)
        pygame.draw.rect(screen, color3, button3_rect, border_radius=10)
        text3 = font.render("Train 10 Epochs", True, (255, 255, 255))
        screen.blit(text3, (button_x + 100, 260))
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
            if event.type == pygame.MOUSEBUTTONDOWN:
                if button1_rect.collidepoint(event.pos):
                    pygame.quit()
                    if not load_trained_model():
                        print("Training 10 epochs since no model exists...")
                        train_model(10)
                    return 'use_trained'
                elif button2_rect.collidepoint(event.pos):
                    pygame.quit()
                    train_model(1)
                    return 'train_1'
                elif button3_rect.collidepoint(event.pos):
                    pygame.quit()
                    train_model(10)
                    return 'train_10'
        
        pygame.display.flip()
        clock.tick(60)

# pygame

def draw_digit():
    pygame.init()
    window_size = 280 #28*28 10x
    display_height = window_size + 50
    screen = pygame.display.set_mode((window_size, display_height))
    pygame.display.set_caption("Draw a digit")
    clock = pygame.time.Clock()
    screen.fill((0, 0, 0))
    drawin = False
    predictions = None
    has_seen_prediction = False

    font = pygame.font.Font(None, 36)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Auto-clear if user is starting a new drawing after seeing a prediction
                if predictions is not None and has_seen_prediction:
                    screen.fill((0, 0, 0))
                    predictions = None
                    has_seen_prediction = False
                drawin = True
            if event.type == pygame.MOUSEBUTTONUP:
                drawin = False
                predictions = predict_digit(screen)
                has_seen_prediction = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    screen.fill((0, 0, 0))
                    predictions = None
                    has_seen_prediction = False
            if event.type == pygame.MOUSEMOTION and drawin:
                pygame.draw.circle(screen, (255, 255, 255), event.pos, 8)

        if predictions is not None:
                text = font.render(f"Prediction: {predictions}", True, (0, 255, 0))
                screen.blit(text, (10, window_size + 10))
            
        pygame.display.flip()
        clock.tick(60)

#Processing the drawing
def process_drawing(screen):
    surface = pygame.surfarray.array3d(screen)
    gray = np.dot(surface[..., :3], [0.2989, 0.587, 0.114])
    gray = np.transpose(gray, (1, 0))
    gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    gray = gray.astype(np.float32) / 255.0
    gray = (gray -0.5) / 0.5
    tensor = torch.tensor(gray, dtype = torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor

def predict_digit(screen):
    image = process_drawing(screen)
    if image is None:
        return None
    
    model.eval()
    with torch.no_grad():
        output=model(image)
        _, prediction = torch.max(output, 1)
    return prediction.item()

# Main execution
if __name__ == "__main__":
    choice = show_menu()
    if choice is not None:
        draw_digit()