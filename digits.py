import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pygame
import numpy as np
import cv2

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
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

model = ANN()
loss_fn = nn.CrossEntropyLoss() #loss is y direction of the graph, weight is x direction, its a curve and the coordinates at (weight - x points) x is calculated, gradient is calculated later
optimizer = optim.Adam(model.parameters(), lr=0.001) #lr=learning rate, controls how much the optimizer should change the parameters, if its too large, the changes may be too large, and also there is no one best rate, you usually test between 0.1, 0.01, 0.001 to find which is the best

# Training

episodes = 10

for epoch in range(episodes):  #epoch is each episode
    running_loss = 0.0 #resetting the loss, otherwise it would accumulate
    for images, labels in trainloader: 
        optimizer.zero_grad() #resetting the gradients, same as before
        outputs = model(images) #get the model's predictions
        loss = loss_fn(outputs, labels) #calculate the losses, by comparing with the real value in "labels"
        loss.backward() #here we calculate the gradient of the losses respective to the weight, bias etc, by backtracking to weight and bias, hence the name
        optimizer.step() #run the optimizer
        running_loss +=loss.item()
    print(f"Epoch {epoch+1}, loss: {running_loss/len(trainloader):.4f}")

# Testing the model
correct = 0
total = 0

model.eval()

with torch.no_grad():
    for images, labels in testloader:
        outputs = model(images)
        _, predictions = torch.max(outputs, 1) #torch max returns the max prob with the index
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

print(f"Accuracy: {100*correct/total:.2f}%")

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

    font = pygame.font.Font(None, 36)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.MOUSEBUTTONDOWN:
                drawin = True
            if event.type == pygame.MOUSEBUTTONUP:
                drawin = False
                predictions = predict_digit(screen)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    screen.fill((0, 0, 0))
                    predictions = None
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

draw_digit()