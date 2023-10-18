import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('--architecture', dest="architecture", action="store", default="vgg16", type=str)
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001, type=float)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
    parser.add_argument('--gpu', dest="gpu", action="store_true", default=False)
    args = parser.parse_args()
    return args

def create_transformer(train_dir, test_dir):
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    return train_data, test_data

def create_data_loader(data, batch_size, train=True):
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=train)

def get_device(gpu):
    if gpu and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def create_model(architecture, hidden_units):
    model = getattr(models, architecture)(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.5)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    return model

def validate(model, test_loader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    model.to(device)
    
    with torch.no_grad():
        model.eval()
        
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            test_loss += criterion(output, labels).item()
            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy

def train_model(model, train_loader, valid_loader, device, criterion, optimizer, epochs, print_every):
    steps = 0
    
    model.to(device)
    
    for e in range(epochs):
        running_loss = 0
        model.train()
        
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = validate(model, valid_loader, criterion, device)
                
                print(f"Epoch: {e + 1}/{epochs}",
                      f"Training Loss: {running_loss / print_every:.4f}",
                      f"Validation Loss: {valid_loss / len(valid_loader):.4f}",
                      f"Validation Accuracy: {accuracy / len(valid_loader):.4f}")
                
                running_loss = 0
                model.train()

    return model

def validate_accuracy(model, test_loader, device):
    correct = 0
    total = 0
    
    model.to(device)
    
    with torch.no_grad():
        model.eval()
        
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy on test images: {accuracy:.2f}%')

def save_checkpoint(model, save_dir, train_data):
    if isdir(save_dir):
        model.class_to_idx = train_data.class_to_idx
        
        checkpoint = {
            'architecture': model.name,
            'classifier': model.classifier,
            'class_to_idx': model.class_to_idx,
            'state_dict': model.state_dict()
        }
        
        torch.save(checkpoint, save_dir)
        print(f"Model checkpoint saved at {save_dir}")
    else:
        print("Directory not found, model will not be saved.")

def main():
    args = parse_arguments()
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    
    train_data, test_data = create_transformer(train_dir, test_dir)
    
    train_loader = create_data_loader(train_data, batch_size=64)
    test_loader = create_data_loader(test_data, batch_size=64, train=False)
    
    model = create_model(args.architecture, args.hidden_units)
    
    device = get_device(args.gpu)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    print_every = 40
    
    trained_model = train_model(model, train_loader, test_loader, device, criterion, optimizer, args.epochs, print_every)
    
    print("\nTraining process is completed successfully!")
    
    validate_accuracy(trained_model, test_loader, device)
    
    save_checkpoint(trained_model, args.save_dir, train_data)

if __name__ == '__main__':
    main()
