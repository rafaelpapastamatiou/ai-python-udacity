import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import os

def parse_input_args():
    parser = argparse.ArgumentParser(description="AIPND Training app")

    parser.add_argument("data_dir", type=str, help="Directory that contains the desired datasets")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--arch", type=str, default="vgg19", help="Base architecture to use")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Network Learning Rate")
    parser.add_argument("--hidden_units", type=str, default="4096,102", help="Network Hidden Units separated by comma")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout to use on each hidden layer")
    parser.add_argument("--epochs", type=float, default=10, help="Epochs amount to train the network")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size to use when training model")
    parser.add_argument("--gpu", action='store_true', default=False, help="Use GPU (Cuda) if available")
    parser.add_argument("--print_every", type=int, default=5, help="Validate and print accuracy every X steps when training")
    
    return parser.parse_args()
    
def load_training_data(folder, batch_size=64):
    train_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    
    train_dataset = datasets.ImageFolder(folder + "/train", transform=train_transform)
    valid_dataset = datasets.ImageFolder(folder + "/valid" , transform=valid_transform)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
    
    return (train_dataloader, train_dataset), (valid_dataloader, valid_dataset)

def build_base_model(arch):
    print("Using arch: ", arch)
    
    if hasattr(models, arch) == False:
        raise Exception("Architecture not supported")
        
    return getattr(models, arch)(pretrained=True)

def build_model(base_model, hidden_units, dropout = 0.2):
    hidden_units = [int(hu) for hu in hidden_units.split(",")]
    
    if len(hidden_units) % 2 != 0:
        raise Exception("Hidden units list length must be an even number")
        
    base_model_classifier_key = next(reversed(base_model._modules))
    
    base_classifier_layer = base_model._modules[base_model_classifier_key][0]
    
    additional_layers = []
    
    for i, hu in enumerate(hidden_units):
        if i % 2 != 0:
            additional_layers.append(nn.ReLU())
            additional_layers.append(nn.Dropout(dropout))
            additional_layers.append(
                nn.Linear(hidden_units[i - 1], hu)
            )
    
    
    classifier = nn.Sequential(
        base_classifier_layer,
        *additional_layers,
        nn.LogSoftmax(dim=1)
    )
    
    base_model._modules[base_model_classifier_key] = classifier
    
    return base_model, classifier

def train_model(
    model,
    classifier,
    train_dataloader,
    valid_dataloader,
    class_to_idx,
    epochs=10,
    learning_rate=0.01,
    print_every=5,
    gpu=False,
):
    device = torch.device("cuda" if gpu == True and torch.cuda.is_available() else "cpu")
    
    print("Traning on", device)
    
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=learning_rate)
    
    model.to(device)
    
    steps = 0
    
    for epoch in range(epochs):
        running_loss = 0
        
        for images, labels in train_dataloader:
            steps += 1
            
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()
                
                with torch.no_grad():
                    for images, labels in valid_dataloader:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        loss = criterion(logps, labels)
                        validation_loss += loss.item()
                        ps = torch.exp(logps)
                        _, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(
                    f"Epoch {epoch+1}/{epochs}.. "
                    f"Training loss: {running_loss/print_every:.3f}.. "
                    f"Validation loss: {validation_loss/len(valid_dataloader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(valid_dataloader):.3f}"
                )

                running_loss = 0
                model.train()
    
    model.class_to_idx = class_to_idx
    print("Traning finished!")

def save_model(model, classifier, arch, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_filepath = checkpoint_dir + "/" + arch + "_checkpoint.pth"
    
    checkpoint = {
        'arch': arch,
        'class_to_idx': model.class_to_idx,
        'classifier': classifier,
        'state_dict': model.state_dict(),
    }

    torch.save(checkpoint, checkpoint_filepath)
                
def main():
    args = parse_input_args()
    
    (train_dataloader, train_dataset), (valid_dataloader, _) = load_training_data(args.data_dir)
    
    base_model = build_base_model(args.arch)
    
    model, classifier = build_model(base_model, args.hidden_units, args.dropout)
    
    train_model(
        model,
        classifier,
        train_dataloader,
        valid_dataloader,
        train_dataset.class_to_idx,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        print_every=args.print_every,
        gpu=args.gpu
    )
    
    save_model(model, classifier, args.arch, args.save_dir)
    
if __name__ == "__main__":
    main()
