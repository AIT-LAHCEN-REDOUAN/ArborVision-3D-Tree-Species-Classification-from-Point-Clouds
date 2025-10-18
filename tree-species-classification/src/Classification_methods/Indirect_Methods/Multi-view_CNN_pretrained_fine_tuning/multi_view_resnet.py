import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score

# Import the MultiViewPCDataset directly
from multi_view import MultiViewPCDataset, SPECIES

SPECIES = ["Buche", "Douglasie", "Eiche", "Esche", "Fichte", "Kiefer", "Roteiche"]

class MultiViewResNet18(nn.Module):
    """
    ResNet18-based model for multi-view 3D point cloud classification.
    Takes multiple views of a 3D point cloud and aggregates features using view pooling.
    """
    def __init__(self, num_classes=len(SPECIES), pretrained=True, view_pooling='max'):
        """
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained ImageNet weights
            view_pooling (str): Method for aggregating features from multiple views
                                ('max', 'avg', or 'attention')
        """
        super(MultiViewResNet18, self).__init__()
        
        # Load pretrained ResNet18 model
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.base_model = resnet18(weights=weights)
        
        # Modify first conv layer to accept grayscale input (1 channel)
        # If using pretrained weights, we need to adapt the weights
        if pretrained:
            # Get the original weights
            original_weight = self.base_model.conv1.weight.data
            # Create new conv layer with 1 input channel
            self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Average the weights across the RGB channels and copy to new conv
            self.base_model.conv1.weight.data = original_weight.mean(dim=1, keepdim=True)
        else:
            # Simply replace with a new conv layer
            self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Get the feature dimension before the final classification layer
        self.feature_dim = self.base_model.fc.in_features
        
        # Replace the final fully connected layer
        self.base_model.fc = nn.Identity()  # Remove the final FC layer
        
        # Add a new classifier after view pooling
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
        # View pooling method
        self.view_pooling = view_pooling
        
        # Attention mechanism for view pooling if using 'attention'
        if view_pooling == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(self.feature_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, num_views, channels, height, width]
        Returns:
            Class logits of shape [batch_size, num_classes]
        """
        batch_size, num_views, channels, height, width = x.size()
        
        # Reshape to process all views
        x = x.view(batch_size * num_views, channels, height, width)
        
        # Extract features from base model (without final classification)
        features = self.base_model(x)  # [batch_size * num_views, feature_dim]
        
        # Reshape back to separate views
        features = features.view(batch_size, num_views, self.feature_dim)  # [batch_size, num_views, feature_dim]
        
        # Apply view pooling
        if self.view_pooling == 'max':
            # Max pooling across views
            pooled_features, _ = torch.max(features, dim=1)  # [batch_size, feature_dim]
        
        elif self.view_pooling == 'avg':
            # Average pooling across views
            pooled_features = torch.mean(features, dim=1)  # [batch_size, feature_dim]
        
        elif self.view_pooling == 'attention':
            # Attention-weighted pooling
            attention_scores = self.attention(features.view(-1, self.feature_dim)).view(batch_size, num_views, 1)
            attention_weights = F.softmax(attention_scores, dim=1)
            pooled_features = torch.sum(features * attention_weights, dim=1)  # [batch_size, feature_dim]
        
        else:
            raise ValueError(f"Unknown view pooling method: {self.view_pooling}")
        
        # Final classification
        logits = self.classifier(pooled_features)
        
        return logits

    def count_parameters(self):
        """
        Count the number of trainable parameters in the model
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def evaluate_model(model, data_loader, device='cuda'):
    """
    Evaluate the model using comprehensive metrics
    
    Args:
        model: MultiViewResNet18 model
        data_loader: DataLoader for evaluation data
        device: Device to evaluate on ('cuda' or 'cpu')
    
    Returns:
        Dictionary containing evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    all_labels = []
    all_predictions = []
    execution_time = 0
    sample_count = 0
    
    with torch.no_grad():
        for views, labels, _ in data_loader:
            views, labels = views.to(device), labels.to(device)
            
            # Measure execution time
            start_time = time.time()
            outputs = model(views)
            end_time = time.time()
            
            execution_time += (end_time - start_time)
            sample_count += views.size(0)
            
            _, predicted = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Calculate metrics
    metrics = {
        'overall_accuracy': accuracy_score(all_labels, all_predictions),
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_predictions),
        'f1_score': f1_score(all_labels, all_predictions, average='weighted'),
        'precision_per_class': precision_score(all_labels, all_predictions, average=None, zero_division=0).tolist(),
        'avg_execution_time_per_sample': execution_time / sample_count if sample_count > 0 else 0,
        'num_parameters': model.count_parameters()
    }
    
    return metrics


def print_evaluation_report(metrics, species_names=SPECIES):
    """
    Print a formatted evaluation report
    
    Args:
        metrics: Dictionary containing evaluation metrics
        species_names: List of species names for class-wise metrics
    """
    print("\n" + "=" * 50)
    print("EVALUATION METRICS")
    print("=" * 50)
    
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"F1 Score (Weighted): {metrics['f1_score']:.4f}")
    
    print("\nPrecision per Class:")
    for i, species in enumerate(species_names):
        print(f"  {species}: {metrics['precision_per_class'][i]:.4f}")
    
    print("\nModel Performance:")
    print(f"  Average Execution Time per Sample: {metrics['avg_execution_time_per_sample']*1000:.2f} ms")
    print(f"  Number of Parameters: {metrics['num_parameters']:,}")
    print("=" * 50)


# Simple test to make sure the model works
if __name__ == "__main__":
    # Create a dummy input
    batch_size = 2
    num_views = 36  # 3 elevations x 12 azimuths
    channels = 1    # Grayscale images
    height = 224
    width = 224
    
    # Create random input tensor
    x = torch.randn(batch_size, num_views, channels, height, width)
    
    # Create model
    model = MultiViewResNet18(num_classes=len(SPECIES), pretrained=False)
    
    # Print model summary
    print(f"Model created successfully with {model.count_parameters():,} parameters")
    
    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")


def train_multi_view_model(model, train_loader, val_loader=None, num_epochs=30, 
                          learning_rate=0.001, weight_decay=1e-4, device='cuda'):
    """
    Train the multi-view model
    
    Args:
        model: MultiViewResNet18 model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        device: Device to train on ('cuda' or 'cpu')
    
    Returns:
        Trained model and training history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Use different learning rates for pretrained layers and new layers
    # This is a common fine-tuning strategy
    base_params = list(model.base_model.parameters())
    classifier_params = list(model.classifier.parameters())
    
    if model.view_pooling == 'attention':
        classifier_params += list(model.attention.parameters())
    
    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': learning_rate * 0.1},  # Lower learning rate for pretrained layers
        {'params': classifier_params, 'lr': learning_rate}   # Higher learning rate for new layers
    ], weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=3)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for views, labels, _ in train_loader:
            views, labels = views.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(views)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * views.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            
            with torch.no_grad():
                for views, labels, _ in val_loader:
                    views, labels = views.to(device), labels.to(device)
                    outputs = model(views)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * views.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Update learning rate based on validation loss
            scheduler.step(val_loss)
            
            print(f'Epoch {epoch+1}/{num_epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        else:
            print(f'Epoch {epoch+1}/{num_epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
    
    return model, history


def create_data_loaders(dataset_path, batch_size=16, num_workers=4, test_split=0.2):
    """
    Create data loaders for training and validation
    
    Args:
        dataset_path: Path to the dataset
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        test_split: Fraction of data to use for validation
    
    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader, random_split
    from multi_view import MultiViewPCDataset  # Changed from relative import to direct import
    
    # Create dataset
    dataset = MultiViewPCDataset(root=dataset_path)
    
    # Split into train and validation sets
    val_size = int(len(dataset) * test_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader


if __name__ == "__main__":
    import argparse
    import os
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Train MultiViewResNet18 for tree species classification')
    parser.add_argument('--data_path', type=str, default="D:\\github\\dataverse_files",
                        help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--view_pooling', type=str, default='max', 
                        choices=['max', 'avg', 'attention'], help='View pooling method')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--save_dir', type=str, default="D:\\github\\tree-species-classification\\models",
                        help='Directory to save the model')
    
    args = parser.parse_args()
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(args.data_path, args.batch_size)
    
    # Create model
    model = MultiViewResNet18(num_classes=len(SPECIES), pretrained=args.pretrained, 
                             view_pooling=args.view_pooling)
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, history = train_multi_view_model(model, train_loader, val_loader, 
                                          num_epochs=args.num_epochs,
                                          learning_rate=args.learning_rate,
                                          device=device)
    
    # Save model
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.save_dir, 
                            f"multi_view_resnet18_{args.view_pooling}_{timestamp}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'species': SPECIES,
        'history': history
    }, save_path)
    
    print(f"Model saved to {save_path}")