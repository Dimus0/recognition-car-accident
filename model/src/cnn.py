import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from tqdm import tqdm

class AccidentCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(128*28*28,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,2)
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
    

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder("dataset_cnn/train",transform=transform)
    test_dataset = datasets.ImageFolder("dataset_cnn/test",transform=transform)

    train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=16,shuffle=True)


    model = AccidentCNN()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(torch.cuda.is_available())

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    train_loss = []
    test_loss = []
    train_accs = []
    test_accs = []

    # Змінити на довільну кількість
    EPOCHS = 15

    # tqdm doesn't worked need to change for stable functionality it's a RANGE
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        correct = 0
        total = 0
        running_loss = 0

        for images,labels in train_loader:

            images,labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()


            running_loss += loss.item()
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()


        train_loss.append(running_loss / len(train_loader))
        train_accs.append(correct / total)


        model.eval()
        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for images,labels in test_loader:
                images,labels = images.to(device),labels.to(device)

                outputs = model(images)
                loss = criterion(outputs,labels)

                val_loss += loss.item()
                _,predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()


        test_loss.append(val_loss / len(test_loader))
        test_accs.append(correct / total)


        print(f"Epoch {epoch+1}/{EPOCHS} | Train loss: {train_loss[-1]:.4f} | Val Acc: {test_accs[-1]:.4f}")


    torch.save(model.state_dict(),"accident_cnn_model")
    print("Модель збережена")



    # Printed plots
    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for images,labels in test_loader:
            images = images.to(device)
            outputs = model(images)

            preds = outputs.argmax(dim=1).cpu().numpy()
            pred_labels.extend(preds)
            true_labels.extend(labels.numpy())

    print(classification_report(true_labels,pred_labels,target_names=train_dataset.classes))


    cm = confusion_matrix(true_labels,pred_labels)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm,annot=True, fmt="d",cmap="Blues", xticklabels=train_dataset.classes,yticklabels=train_dataset.classes)
    plt.title("Confusing Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("confusing_matrix.png")


    # Training Curves
    plt.figure(figsize=(12,6))
    plt.plot(train_loss,label="Train Loss")
    plt.plot(test_loss,label="Test Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig("training_loss.png")

    plt.figure(figsize=(12,6))
    plt.plot(train_accs,label="Train Accuracy")
    plt.plot(test_accs,label="Test Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.savefig("training_accuracy_curve.png")


