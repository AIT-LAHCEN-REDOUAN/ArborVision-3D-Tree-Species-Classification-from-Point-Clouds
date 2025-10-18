import torch
import torch.nn as nn
import os
import argparse
import time
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt

# Import from existing modules
from multi_view import MultiViewPCDataset, SPECIES
from multi_view_resnet import evaluate_model, print_evaluation_report

# Debug flag
DEBUG = True

class MultiViewResNet50(nn.Module):
    """
    ResNet50-based model for multi-view 3D point cloud classification.
    This is a more powerful backbone compared to ResNet18 for fine-tuning.
    """
    def __init__(self, num_classes=len(SPECIES), pretrained=True, view_pooling='max'):
        """
        Args:
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained ImageNet weights
            view_pooling (str): Method for aggregating features from multiple views
                                ('max', 'avg', or 'attention')
        """
        super(MultiViewResNet50, self).__init__()
        
        # Load pretrained ResNet50 model
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.base_model = resnet50(weights=weights)
        
        # Modify first conv layer to accept grayscale input (1 channel)
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
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # View pooling method
        self.view_pooling = view_pooling
        
        # Attention mechanism for view pooling if using 'attention'
        if view_pooling == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(self.feature_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
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
            attention_weights = torch.softmax(attention_scores, dim=1)
            pooled_features = torch.sum(features * attention_weights, dim=1)  # [batch_size, feature_dim]
            
            # Debug: visualize attention weights if needed
            if DEBUG and torch.rand(1).item() < 0.01:  # Only log occasionally to avoid flooding
                print(f"DEBUG: Attention weights shape: {attention_weights.shape}")
                print(f"DEBUG: Attention weights sample: {attention_weights[0].detach().cpu().numpy()}")
        
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


def fine_tune_model(model, train_loader, val_loader=None, num_epochs=50, 
                   learning_rate=0.0005, weight_decay=1e-5, device='cuda',
                   patience=10, debug_dir=None):
    """
    Fine-tune the multi-view model with specialized learning rates and schedulers
    
    Args:
        model: MultiViewResNet50 model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        num_epochs: Number of training epochs
        learning_rate: Base learning rate for optimizer
        weight_decay: Weight decay for regularization
        device: Device to train on ('cuda' or 'cpu')
        patience: Number of epochs to wait for improvement before early stopping
        debug_dir: Directory to save debug information
    
    Returns:
        Trained model and training history
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Create debug directory if needed
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)
    
    # Group parameters for different learning rates
    # 1. Base model early layers (lowest learning rate)
    # 2. Base model later layers (medium learning rate)
    # 3. New classifier layers (highest learning rate)
    
    # Get all layers from base model except the last 2 blocks
    early_layers = []
    for name, param in model.base_model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            early_layers.append(param)
    
    # Get the last 2 blocks of base model
    later_layers = []
    for name, param in model.base_model.named_parameters():
        if 'layer4' in name and 'fc' not in name:
            later_layers.append(param)
    
    # Get classifier parameters
    classifier_params = list(model.classifier.parameters())
    
    # Add attention parameters to classifier group if using attention pooling
    if model.view_pooling == 'attention':
        classifier_params += list(model.attention.parameters())
    
    # Create parameter groups with different learning rates
    optimizer = torch.optim.AdamW([
        {'params': early_layers, 'lr': learning_rate * 0.1},      # Lowest learning rate
        {'params': later_layers, 'lr': learning_rate * 0.5},     # Medium learning rate
        {'params': classifier_params, 'lr': learning_rate}       # Highest learning rate
    ], weight_decay=weight_decay)
    
    # Learning rate scheduler with warmup and cosine annealing
    def lr_lambda(epoch):
        # Warmup for first 5 epochs
        if epoch < 5:
            return epoch / 5
        # Cosine annealing for remaining epochs
        return 0.5 * (1 + torch.cos(torch.tensor((epoch - 5) / (num_epochs - 5) * torch.pi)).item())
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 
        'lr': [], 'epoch_times': [], 'best_epoch': 0
    }
    
    best_val_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0
    
    # Debug: Print model structure if debug mode is on
    if DEBUG:
        print("\nDEBUG: Model Structure:")
        for name, param in model.named_parameters():
            print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")
        print(f"\nDEBUG: Total parameters: {model.count_parameters():,}")
    
    print(f"\nStarting fine-tuning for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        # Create progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", 
                         leave=False, ncols=100)
        
        batch_losses = []
        for views, labels, _ in train_pbar:
            views, labels = views.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(views)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            train_loss += batch_loss * views.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            batch_correct = predicted.eq(labels).sum().item()
            train_correct += batch_correct
            
            # Update progress bar
            batch_acc = batch_correct / labels.size(0)
            train_pbar.set_postfix({
                'loss': f"{batch_loss:.4f}", 
                'acc': f"{batch_acc:.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Store current learning rate
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Debug: Print batch loss statistics
        if DEBUG and batch_losses:
            print(f"DEBUG: Batch losses - min: {min(batch_losses):.4f}, max: {max(batch_losses):.4f}, "
                  f"mean: {np.mean(batch_losses):.4f}, std: {np.std(batch_losses):.4f}")
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            all_preds, all_labels = [], []
            
            # Create progress bar for validation
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", 
                           leave=False, ncols=100)
            
            with torch.no_grad():
                for views, labels, _ in val_pbar:
                    views, labels = views.to(device), labels.to(device)
                    outputs = model(views)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * views.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    batch_correct = predicted.eq(labels).sum().item()
                    val_correct += batch_correct
                    
                    # Store predictions and labels for metrics calculation
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    # Update progress bar
                    batch_acc = batch_correct / labels.size(0)
                    val_pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{batch_acc:.4f}"})
            
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Calculate additional metrics for debugging
            if DEBUG:
                balanced_acc = balanced_accuracy_score(all_labels, all_preds)
                f1 = f1_score(all_labels, all_preds, average='weighted')
                precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
                
                print(f"\nDEBUG: Validation Metrics for Epoch {epoch+1}:")
                print(f"  Balanced Accuracy: {balanced_acc:.4f}")
                print(f"  F1 Score (Weighted): {f1:.4f}")
                print("  Precision per Class:")
                for i, species in enumerate(SPECIES):
                    print(f"    {species}: {precision_per_class[i]:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                history['best_epoch'] = epoch + 1
                epochs_without_improvement = 0
                print(f"\n[INFO] New best model found! Validation accuracy: {val_acc:.4f}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"\n[INFO] Early stopping triggered after {epoch+1} epochs. "
                          f"No improvement for {patience} epochs.")
                    break
            
            # Calculate epoch time and store it
            epoch_time = time.time() - epoch_start_time
            history['epoch_times'].append(epoch_time)
            
            print(f'Epoch {epoch+1}/{num_epochs} [{epoch_time:.1f}s]: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Debug: Plot learning curves every 5 epochs if debug directory is provided
            if debug_dir is not None and (epoch + 1) % 5 == 0:
                plot_learning_curves(history, os.path.join(debug_dir, f'learning_curves_epoch_{epoch+1}.png'))
        else:
            # Calculate epoch time and store it
            epoch_time = time.time() - epoch_start_time
            history['epoch_times'].append(epoch_time)
            
            print(f'Epoch {epoch+1}/{num_epochs} [{epoch_time:.1f}s]: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Update learning rate
        scheduler.step()
    
    # Load best model if we found one
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n[INFO] Loaded best model from epoch {history['best_epoch']}")
    
    return model, history


def create_data_loaders(dataset_path, batch_size=8, num_workers=4, test_split=0.2):
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
    # Create dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = MultiViewPCDataset(root=dataset_path)
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Split into train and validation sets
    val_size = int(len(dataset) * test_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Split dataset into {train_size} training and {val_size} validation samples")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader


def plot_learning_curves(history, save_path=None):
    """
    Plot learning curves from training history
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(15, 10))
    
    # Plot accuracy
    plt.subplot(2, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    if 'val_acc' in history and history['val_acc']:
        plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(2, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate
    plt.subplot(2, 2, 3)
    plt.plot(history['lr'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    
    # Plot epoch times
    if 'epoch_times' in history and history['epoch_times']:
        plt.subplot(2, 2, 4)
        plt.plot(history['epoch_times'])
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.title('Epoch Execution Times')
        plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Learning curves saved to {save_path}")
    else:
        plt.show()


def evaluate_model_comprehensive(model, data_loader, device='cuda'):
    """
    Evaluate the model using comprehensive metrics
    
    Args:
        model: MultiViewResNet50 model
        data_loader: DataLoader for evaluation data
        device: Device to evaluate on ('cuda' or 'cpu')
    
    Returns:
        Dictionary containing evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    all_labels = []
    all_predictions = []
    execution_times = []
    sample_count = 0
    
    # Create progress bar for evaluation
    eval_pbar = tqdm(data_loader, desc="Evaluating", leave=True, ncols=100)
    
    with torch.no_grad():
        for views, labels, _ in eval_pbar:
            views, labels = views.to(device), labels.to(device)
            
            # Measure execution time
            start_time = time.time()
            outputs = model(views)
            end_time = time.time()
            
            batch_time = end_time - start_time
            execution_times.append(batch_time)
            sample_count += views.size(0)
            
            _, predicted = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
            # Update progress bar
            eval_pbar.set_postfix({'time/batch': f"{batch_time:.4f}s"})
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    # Calculate metrics
    metrics = {
        'overall_accuracy': accuracy_score(all_labels, all_predictions),
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_predictions),
        'f1_score': f1_score(all_labels, all_predictions, average='weighted'),
        'f1_score_per_class': f1_score(all_labels, all_predictions, average=None).tolist(),
        'precision_per_class': precision_score(all_labels, all_predictions, average=None, zero_division=0).tolist(),
        'avg_execution_time_per_sample': sum(execution_times) / sample_count if sample_count > 0 else 0,
        'avg_execution_time_per_batch': sum(execution_times) / len(execution_times) if execution_times else 0,
        'num_parameters': model.count_parameters(),
        'confusion_matrix': conf_matrix.tolist()
    }
    
    return metrics


def print_comprehensive_evaluation_report(metrics, species_names=SPECIES):
    """
    Print a formatted comprehensive evaluation report
    
    Args:
        metrics: Dictionary containing evaluation metrics
        species_names: List of species names for class-wise metrics
    """
    print("\n" + "=" * 60)
    print("COMPREHENSIVE EVALUATION METRICS")
    print("=" * 60)
    
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"F1 Score (Weighted): {metrics['f1_score']:.4f}")
    
    print("\nPrecision per Class:")
    for i, species in enumerate(species_names):
        print(f"  {species}: {metrics['precision_per_class'][i]:.4f}")
    
    print("\nF1 Score per Class:")
    for i, species in enumerate(species_names):
        print(f"  {species}: {metrics['f1_score_per_class'][i]:.4f}")
    
    print("\nModel Performance:")
    print(f"  Average Execution Time per Sample: {metrics['avg_execution_time_per_sample']*1000:.2f} ms")
    print(f"  Average Execution Time per Batch: {metrics['avg_execution_time_per_batch']*1000:.2f} ms")
    print(f"  Number of Parameters: {metrics['num_parameters']:,}")
    
    print("\nConfusion Matrix:")
    conf_matrix = np.array(metrics['confusion_matrix'])
    # Print header
    header = "    " + "".join(f"{species[:3]:>5}" for species in species_names)
    print(header)
    
    # Print rows
    for i, row in enumerate(conf_matrix):
        row_str = f"{species_names[i][:3]:>3} " + "".join(f"{val:>5}" for val in row)
        print(row_str)
    
    print("=" * 60)


def plot_confusion_matrix(metrics, species_names=SPECIES, save_path=None):
    """
    Plot confusion matrix from evaluation metrics
    
    Args:
        metrics: Dictionary containing evaluation metrics with confusion_matrix
        species_names: List of species names
        save_path: Path to save the plot (optional)
    """
    conf_matrix = np.array(metrics['confusion_matrix'])
    
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(len(species_names))
    plt.xticks(tick_marks, species_names, rotation=45)
    plt.yticks(tick_marks, species_names)
    
    # Add text annotations
    thresh = conf_matrix.max() / 2
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune MultiViewResNet50 for tree species classification')
    parser.add_argument('--data_path', type=str, default="D:\\github\\dataverse_files",
                        help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--view_pooling', type=str, default='attention', 
                        choices=['max', 'avg', 'attention'], help='View pooling method')
    parser.add_argument('--save_dir', type=str, default="D:\\github\\tree-species-classification\\models",
                        help='Directory to save the model')
    parser.add_argument('--debug_dir', type=str, default="D:\\github\\tree-species-classification\\debug",
                        help='Directory to save debug information')
    parser.add_argument('--patience', type=int, default=10, 
                        help='Number of epochs to wait for improvement before early stopping')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Set debug flag
    DEBUG = args.debug  # Remove the global declaration
    
    # Create debug directory
    if DEBUG:
        os.makedirs(args.debug_dir, exist_ok=True)
        print(f"Debug mode enabled. Debug information will be saved to {args.debug_dir}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(args.data_path, args.batch_size)
    
    # Create model - always use pretrained weights for fine-tuning
    model = MultiViewResNet50(num_classes=len(SPECIES), pretrained=True, 
                            view_pooling=args.view_pooling)
    
    print(f"Created ResNet50 model with {model.count_parameters():,} parameters")
    print(f"Using {args.view_pooling} pooling for multi-view aggregation")
    
    # Fine-tune model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    model, history = fine_tune_model(model, train_loader, val_loader, 
                                   num_epochs=args.num_epochs,
                                   learning_rate=args.learning_rate,
                                   device=device,
                                   patience=args.patience,
                                   debug_dir=args.debug_dir if DEBUG else None)
    
    # Plot final learning curves
    if DEBUG:
        plot_learning_curves(history, os.path.join(args.debug_dir, 'final_learning_curves.png'))
    
    # Evaluate model with comprehensive metrics
    print("\nEvaluating model on validation set...")
    metrics = evaluate_model_comprehensive(model, val_loader, device=device)
    print_comprehensive_evaluation_report(metrics)
    
    # Plot confusion matrix
    if DEBUG:
        plot_confusion_matrix(metrics, save_path=os.path.join(args.debug_dir, 'confusion_matrix.png'))
    
    # Save model
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.save_dir, 
                            f"multi_view_resnet50_finetuned_{args.view_pooling}_{timestamp}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'species': SPECIES,
        'history': history,
        'metrics': metrics
    }, save_path)
    
    print(f"\nModel saved to {save_path}")