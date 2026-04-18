#main.py

import torch
import os
import sys
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.append(os.path.abspath("image-classifier"))

from models.model import CIFAR10CNN
from train.train import train_model
from utils.helper import evaluate_model, plot_training_history
from dataset.custom_dataset import CIFAR10Dataset 

def run_experiments():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiments = [
        {"epochs": 5,  "opt": "SGD",  "lr": 0.01,  "batch": 32},
        {"epochs": 10, "opt": "Adam", "lr": 0.001, "batch": 64},
        {"epochs": 20, "opt": "Adam", "lr": 0.001, "batch": 64}
    ]

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    best_acc = 0.0

    for i, exp in enumerate(experiments):
        print(f"\n>>> RUNNING EXPERIMENT {i+1} <<<")
        train_set = CIFAR10Dataset(root_dir='/content/final_dataset/train', transform=transform)
        test_set = CIFAR10Dataset(root_dir='/content/final_dataset/test', transform=transform)
        
        train_loader = DataLoader(train_set, batch_size=exp['batch'], shuffle=True)
        test_loader = DataLoader(test_set, batch_size=exp['batch'], shuffle=False)
        
        model = CIFAR10CNN()
        trained_model, history = train_model(model, train_loader, exp['epochs'], exp['lr'], exp['opt'], device)
        
        criterion = torch.nn.CrossEntropyLoss()
        test_loss, test_acc = evaluate_model(trained_model, test_loader, criterion, device)
        
        print(f"\nEXPERIMENT {i+1} FINAL RESULTS:")
        print(f"Train Accuracy: {history['train_acc'][-1]:.2f}%")
        print(f"Test Accuracy:  {test_acc:.2f}%")
        
        # UPDATED: Save the plot with a unique name for each experiment 
        plot_path = f"image-classifier/plots/exp_{i+1}_results.png"
        plot_training_history(history, save_path=plot_path)
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(trained_model.state_dict(), "image-classifier/results/best_model.pth")

if __name__ == "__main__":
    run_experiments()
