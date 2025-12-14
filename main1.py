import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from transformers import ViTModel, ViTFeatureExtractor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
def get_num_classes(dataset):
    try:
        # Direct method for datasets with `targets` attribute
        return len(torch.unique(torch.tensor(dataset.targets)))
    except AttributeError:
        # Fallback for datasets without `targets` attribute
        labels = [label for _, label in dataset]
        return len(set(labels))
# GNN Module
class GNNModule(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNModule, self).__init__()
        self.gcn1 = GCNConv(in_features, 128)
        self.gcn2 = GCNConv(128, out_features)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = torch.relu(x)
        x = self.gcn2(x, edge_index)
        return x

# Hybrid ViT-GNN Model
# class ViT_GNN(nn.Module):
#     def __init__(self, vit_model_name, num_classes):
#         super(ViT_GNN, self).__init__()
#         self.vit = ViTModel.from_pretrained(vit_model_name)
#         self.gnn = GNNModule(768, 768)  # Assuming ViT outputs 768 features
#         self.fc = nn.Linear(128, num_classes)

#     def forward(self, pixel_values, edge_index, graph_features):
#         vit_outputs = self.vit(pixel_values=pixel_values).last_hidden_state.mean(dim=1)  # ViT features
#         gnn_outputs = self.gnn(graph_features, edge_index)  # GNN features
#         combined_features = vit_outputs + gnn_outputs  # Combine features
#         logits = self.fc(combined_features)
#         return logits

from torch_geometric.nn import GATConv

# Define Graph Attention Module
class GraphAttentionModule(nn.Module):
    def __init__(self, in_features, out_features, num_heads=2):
        super(GraphAttentionModule, self).__init__()
        self.gat1 = GATConv(in_features, out_features, heads=num_heads, concat=True)
        self.gat2 = GATConv(out_features * num_heads, out_features, heads=1, concat=False)
        
    def forward(self, x, edge_index):
        # First GAT Layer
        x = self.gat1(x, edge_index)
        x = torch.relu(x)

        # Second GAT Layer
        x = self.gat2(x, edge_index)
        
        return x

# Define Spatial Pyramid Pooling Module (SPPM)
class SPSSModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPSSModule, self).__init__()

        # Define pooling levels
        self.pool1 = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.pool2 = nn.AdaptiveAvgPool2d(2)  # 2x2 pooling
        self.pool3 = nn.AdaptiveAvgPool2d(4)  # 4x4 pooling

        # Convolution layers for each pooled output
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Final fusion layer
        self.fusion = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x):
        # Apply pooling and convolution at different scales
        p1 = torch.relu(self.conv1(self.pool1(x)))
        p2 = torch.relu(self.conv2(self.pool2(x)))
        p3 = torch.relu(self.conv3(self.pool3(x)))

        # Upsample to match the largest scale
        p1 = nn.functional.interpolate(p1, size=x.shape[2:], mode='bilinear', align_corners=False)
        p2 = nn.functional.interpolate(p2, size=x.shape[2:], mode='bilinear', align_corners=False)
        p3 = nn.functional.interpolate(p3, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate pooled features
        fused = torch.cat([p1, p2, p3], dim=1)

        # Final fusion layer
        return self.fusion(fused)

# Dataset Preparation
def prepclass ViT_GNN(nn.Module):
    def __init__(self, vit_model_name, num_classes):
        super(ViT_GNN, self).__init__()
        self.vit = ViTModel.from_pretrained(vit_model_name)
        self.gnn = GraphAttentionModule(768,128)#GNNModule(768, 128)  # GNN outputs 128 features
        self.gnn_projector = nn.Linear(128, 768)  # Project GNN output to 768
        self.fc = nn.Linear(768, num_classes)

    def forward(self, pixel_values, edge_index, graph_features):
        vit_outputs = self.vit(pixel_values=pixel_values).last_hidden_state.mean(dim=1)  # (batch_size, 768)
        gnn_outputs = self.gnn(graph_features, edge_index)  # (batch_size, 128)
        gnn_outputs = self.gnn_projector(gnn_outputs)  # Align to (batch_size, 768)
        combined_features = vit_outputs + gnn_outputs  # Combine features
        logits = self.fc(combined_features)  # Final classification
        return logits
are_datasets(dataset_name, data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    if dataset_name == "oxford_pet":
        train_dataset = datasets.OxfordIIITPet(root=data_dir, split="trainval", download=True, transform=transform)
        val_dataset = datasets.OxfordIIITPet(root=data_dir, split="test", download=True, transform=transform)
    elif dataset_name == "imagenet1k":
        train_dataset = datasets.ImageNet(root=data_dir, split="train", transform=transform)
        val_dataset = datasets.ImageNet(root=data_dir, split="val", transform=transform)
    elif dataset_name == "cifar100":
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
        val_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# # Training Loop
# def train_model(model, train_loader, val_loader, device, num_epochs=10, lr=1e-4):
#     model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()

#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#         for batch in train_loader:
#             images, labels = batch
#             pixel_values = images.to(device)
#             labels = labels.to(device)
#             optimizer.zero_grad()

#             # Simulate graph data (this should be replaced with actual graph construction logic)
#             graph_features = torch.rand(pixel_values.size(0), 768).to(device)
#             edge_index = torch.randint(0, pixel_values.size(0), (2, 100)).to(device)

#             outputs = model(pixel_values, edge_index, graph_features)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

#         # Validation
#         model.eval()
#         total_correct = 0
#         total_samples = 0
#         with torch.no_grad():
#             for batch in val_loader:
#                 images, labels = batch
#                 pixel_values = images.to(device)
#                 labels = labels.to(device)


#                 # Simulate graph data
#                 graph_features = torch.rand(pixel_values.size(0), 768).to(device)
#                 edge_index = torch.randint(0, pixel_values.size(0), (2, 100)).to(device)

#                 outputs = model(pixel_values, edge_index, graph_features)
#                 preds = outputs.argmax(dim=1)
#                 total_correct += (preds == labels).sum().item()
#                 total_samples += labels.size(0)

#         accuracy = total_correct / total_samples
#         print(f"Validation Accuracy: {accuracy:.4f}")

# # Main Pipeline
# def main():
#     # Configuration
#     dataset_name = "oxford_pet"
#     data_dir = "./data"
#     batch_size = 16

#     vit_model_name = "google/vit-base-patch16-224"
#     num_epochs = 10
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Prepare datasets
#     train_loader, val_loader = prepare_datasets(dataset_name, data_dir, batch_size)
#     num_classes = get_num_classes(train_loader)  # Change based on dataset
#     # Prepare model
#     model = ViT_GNN(vit_model_name, num_classes)

#     # Train model
#     train_model(model, train_loader, val_loader, device, num_epochs)

# if __name__ == "__main__":
#     main()
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader, device, num_epochs=10, lr=1e-4, save_path="best_model.pth"):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_accuracy = 0.0  # To track the best model

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        all_train_labels = []
        all_train_preds = []

        for batch in train_loader:
            # Unpack batch
            images, labels = batch
            pixel_values = images.to(device)
            labels = labels.to(device)

            # Simulate graph data (replace with actual graph generation if available)
            graph_features = torch.rand(pixel_values.size(0), 768).to(device)
            edge_index = torch.randint(0, pixel_values.size(0), (2, 100)).to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values, edge_index, graph_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(torch.argmax(F.softmax(outputs, dim=1), dim=1).cpu().numpy())

        train_loss = total_train_loss / len(train_loader)
        train_accuracy = accuracy_score(all_train_labels, all_train_preds)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        total_val_loss = 0
        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch
                pixel_values = images.to(device)
                labels = labels.to(device)

                # Simulate graph data (replace with actual graph generation if available)
                graph_features = torch.rand(pixel_values.size(0), 768).to(device)
                edge_index = torch.randint(0, pixel_values.size(0), (2, 100)).to(device)

                outputs = model(pixel_values, edge_index, graph_features)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(torch.argmax(F.softmax(outputs, dim=1), dim=1).cpu().numpy())

        val_loss = total_val_loss / len(val_loader)
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}, Val Accuracy:   {val_accuracy:.4f}")

    # Plot training and validation metrics
    plt.figure(figsize=(12, 6))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("/home/muhammad/NewHM/Flowers102figure10.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Best Validation Accuracy: {best_val_accuracy:.4f} (Model saved at {save_path})")

# Example Usage
if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    #ImageNet
    train_dataset = datasets.Flowers102(root="./data", split='train', download=True, transform=transform)
    val_dataset = datasets.Flowers102(root="./data", split='val', download=True, transform=transform)
    # train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    # val_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    #train_dataset = datasets.OxfordIIITPet(root="/home/muhammad/NewHM/data", split="trainval", download=True, transform=transform)
    #val_dataset = datasets.OxfordIIITPet(root="/home/muhammad/NewHM/data", split="test", download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Define the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = get_num_classes(train_dataset)#len(set(train_dataset.targets))
    model = ViT_GNN("google/vit-base-patch16-224", num_classes=num_classes)

    # Train the model
    train_model(model, train_loader, val_loader, device, num_epochs=100, save_path="/home/muhammad/NewHM/Flowers102best_model10.pth")
