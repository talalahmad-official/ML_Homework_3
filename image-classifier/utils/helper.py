#helper.py

import torch
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def plot_training_history(training_history, save_path=None):
    epochs = range(1, len(training_history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))

    # Plot 1: Epochs vs Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_history['train_loss'], label='Train Loss', color='blue')
    plt.title('Epochs vs Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot 2: Epochs vs Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_history['train_acc'], label='Train Accuracy', color='green')
    plt.title('Epochs vs Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Graph saved to: {save_path}")
        
    plt.show()
    plt.close() # Close to free up memory
