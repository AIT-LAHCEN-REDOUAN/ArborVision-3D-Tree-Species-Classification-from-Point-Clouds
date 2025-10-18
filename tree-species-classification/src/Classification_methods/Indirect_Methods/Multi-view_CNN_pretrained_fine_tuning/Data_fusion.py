import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import time
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import from existing modules
from multi_view import MultiViewPCDataset, SPECIES, render_views, read_points, normalize_points
from multi_view_resnet import evaluate_model, print_evaluation_report
from multi_view_resnet_fine_tuning import MultiViewResNet50, fine_tune_model, plot_learning_curves

# Debug flag
DEBUG = True

class DenseSIFTExtractor:
    """
    Dense SIFT feature extractor for point cloud views
    """
    def __init__(self, step_size=5, bin_size=8, num_features=128):
        """
        Args:
            step_size: Pixel step size for dense SIFT extraction
            bin_size: Size of spatial bins for SIFT descriptor
            num_features: Number of features to keep after dimensionality reduction
        """
        self.step_size = step_size
        self.bin_size = bin_size
        self.num_features = num_features
        # Don't initialize SIFT here - will create on demand
    
    def _get_sift(self):
        """Create SIFT detector on demand"""
        return cv2.SIFT_create()
    
    def extract_features(self, image):
        """
        Extract dense SIFT features from a grayscale image
        
        Args:
            image: Grayscale image as numpy array (H, W)
            
        Returns:
            SIFT features as numpy array
        """
        # Ensure image is in uint8 format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        # Create dense grid of keypoints
        kps = [cv2.KeyPoint(x, y, self.bin_size) 
               for y in range(0, image.shape[0], self.step_size) 
               for x in range(0, image.shape[1], self.step_size)]
        
        # Get SIFT detector on demand
        sift = self._get_sift()
        
        # Compute SIFT descriptors
        _, des = sift.compute(image, kps)
        
        if des is None or len(des) == 0:
            # Return zeros if no descriptors found
            return np.zeros((1, 128), dtype=np.float32)
        
        # Aggregate descriptors using VLAD or Fisher Vectors
        # For simplicity, we'll use average pooling here
        features = np.mean(des, axis=0)
        
        # Normalize features
        features = features / (np.linalg.norm(features) + 1e-7)
        
        return features
    
    def extract_multi_view_features(self, views):
        """
        Extract SIFT features from multiple views
        
        Args:
            views: Tensor of shape [num_views, 1, H, W] or [B, num_views, 1, H, W]
            
        Returns:
            SIFT features as tensor
        """
        if views.dim() == 4:  # [num_views, 1, H, W]
            num_views = views.shape[0]
            all_features = []
            
            for v in range(num_views):
                # Convert to numpy and remove channel dimension
                img = views[v, 0].cpu().numpy()
                features = self.extract_features(img)
                all_features.append(features)
            
            # Stack features from all views
            features_array = np.stack(all_features, axis=0)  # [num_views, feature_dim]
            
            # Max pooling across views
            pooled_features = np.max(features_array, axis=0)  # [feature_dim]
            
            return torch.from_numpy(pooled_features).float()
            
        elif views.dim() == 5:  # [B, num_views, 1, H, W]
            batch_size = views.shape[0]
            num_views = views.shape[1]
            all_batch_features = []
            
            for b in range(batch_size):
                all_view_features = []
                
                for v in range(num_views):
                    # Convert to numpy and remove channel dimension
                    img = views[b, v, 0].cpu().numpy()
                    features = self.extract_features(img)
                    all_view_features.append(features)
                
                # Stack features from all views for this batch item
                features_array = np.stack(all_view_features, axis=0)  # [num_views, feature_dim]
                
                # Max pooling across views
                pooled_features = np.max(features_array, axis=0)  # [feature_dim]
                all_batch_features.append(pooled_features)
            
            # Stack features from all batch items
            batch_features = np.stack(all_batch_features, axis=0)  # [B, feature_dim]
            
            return torch.from_numpy(batch_features).float()
        
        else:
            raise ValueError(f"Unexpected view tensor shape: {views.shape}")


class FusionModel(nn.Module):
    """
    Model that fuses CNN features with classical descriptors (Dense SIFT)
    """
    def __init__(self, num_classes=len(SPECIES), pretrained=True, view_pooling='max'):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights for CNN
            view_pooling: Method for aggregating features from multiple views
        """
        super(FusionModel, self).__init__()
        
        # CNN feature extractor (ResNet50)
        self.cnn_extractor = MultiViewResNet50(num_classes=num_classes, 
                                              pretrained=pretrained,
                                              view_pooling=view_pooling)
        
        # Get the feature dimension from CNN before classification
        self.cnn_feature_dim = self.cnn_extractor.feature_dim
        
        # SIFT feature extractor - don't store as instance variable
        # self.sift_extractor = DenseSIFTExtractor()  # Remove this line
        self.sift_feature_dim = 128  # Standard SIFT descriptor size
        
        # Remove the classifier from CNN extractor
        self.cnn_extractor.classifier = nn.Identity()
        
        # Feature fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.cnn_feature_dim + self.sift_feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Final classifier
        self.classifier = nn.Linear(512, num_classes)
        
        # View pooling method
        self.view_pooling = view_pooling
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, num_views, channels, height, width]
        Returns:
            Class logits of shape [batch_size, num_classes]
        """
        batch_size = x.size(0)
        
        # Extract CNN features
        cnn_features = self.cnn_extractor(x)  # [batch_size, cnn_feature_dim]
        
        # Extract SIFT features - create extractor on demand
        sift_extractor = DenseSIFTExtractor()
        sift_features = sift_extractor.extract_multi_view_features(x)  # [batch_size, sift_feature_dim]
        sift_features = sift_features.to(x.device)
        
        # Concatenate features
        combined_features = torch.cat([cnn_features, sift_features], dim=1)  # [batch_size, cnn_feature_dim + sift_feature_dim]
        
        # Apply fusion layer
        fused_features = self.fusion_layer(combined_features)  # [batch_size, 512]
        
        # Final classification
        logits = self.classifier(fused_features)  # [batch_size, num_classes]
        
        return logits
    
    def count_parameters(self):
        """
        Count the number of trainable parameters in the model
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultiViewFusionDataset(Dataset):
    """
    Dataset wrapper that computes and caches SIFT features
    """
    def __init__(self, base_dataset):
        """
        Args:
            base_dataset: Base MultiViewPCDataset instance
        """
        self.base_dataset = base_dataset
        # Don't store the SIFT extractor as an instance variable
        # self.sift_extractor = DenseSIFTExtractor()  # Remove this line
        self.sift_cache = {}
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get data from base dataset
        views, label, filepath = self.base_dataset[idx]
        
        return views, label, filepath


def train_fusion_model(model, train_loader, val_loader=None, num_epochs=50, 
                      learning_rate=0.0005, weight_decay=1e-5, device='cuda',
                      patience=10, debug_dir=None):
    """
    Train the fusion model
    
    Args:
        model: FusionModel instance
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
    # 1. CNN extractor (lowest learning rate)
    # 2. SIFT and fusion layers (highest learning rate)
    cnn_params = list(model.cnn_extractor.parameters())
    fusion_params = list(model.fusion_layer.parameters()) + list(model.classifier.parameters())
    
    # Create parameter groups with different learning rates
    optimizer = torch.optim.AdamW([
        {'params': cnn_params, 'lr': learning_rate * 0.1},      # Lowest learning rate
        {'params': fusion_params, 'lr': learning_rate}       # Highest learning rate
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
    
    print(f"\nStarting fusion model training for {num_epochs} epochs...")
    
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


def evaluate_fusion_model(model, data_loader, device='cuda'):
    """
    Evaluate the fusion model using comprehensive metrics
    
    Args:
        model: FusionModel instance
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


def create_fusion_data_loaders(dataset_path, batch_size=8, num_workers=4, test_split=0.2):
    """
    Create data loaders for training and validation with fusion dataset
    
    Args:
        dataset_path: Path to the dataset
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        test_split: Fraction of data to use for validation
    
    Returns:
        train_loader, val_loader
    """
    # Create base dataset
    print(f"Loading dataset from {dataset_path}...")
    base_dataset = MultiViewPCDataset(root=dataset_path)
    print(f"Dataset loaded with {len(base_dataset)} samples")
    
    # Wrap with fusion dataset
    fusion_dataset = MultiViewFusionDataset(base_dataset)
    
    # Split into train and validation sets
    val_size = int(len(fusion_dataset) * test_split)
    train_size = len(fusion_dataset) - val_size
    train_dataset, val_dataset = random_split(fusion_dataset, [train_size, val_size])
    print(f"Split dataset into {train_size} training and {val_size} validation samples")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader


def plot_fusion_comparison(cnn_metrics, fusion_metrics, save_path=None):
    """
    Plot comparison between CNN-only and fusion model
    
    Args:
        cnn_metrics: Dictionary containing CNN model metrics
        fusion_metrics: Dictionary containing fusion model metrics
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy comparison
    plt.subplot(2, 2, 1)
    metrics = ['overall_accuracy', 'balanced_accuracy', 'f1_score']
    labels = ['Overall Accuracy', 'Balanced Accuracy', 'F1 Score']
    cnn_values = [cnn_metrics[m] for m in metrics]
    fusion_values = [fusion_metrics[m] for m in metrics]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, cnn_values, width, label='CNN Only')
    plt.bar(x + width/2, fusion_values, width, label='CNN + SIFT Fusion')
    
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Performance Comparison')
    plt.xticks(x, labels)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot per-class precision
    plt.subplot(2, 2, 2)
    x = np.arange(len(SPECIES))
    plt.bar(x - width/2, cnn_metrics['precision_per_class'], width, label='CNN Only')
    plt.bar(x + width/2, fusion_metrics['precision_per_class'], width, label='CNN + SIFT Fusion')
    
    plt.xlabel('Species')
    plt.ylabel('Precision')
    plt.title('Per-Class Precision')
    plt.xticks(x, SPECIES, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Plot execution time
    plt.subplot(2, 2, 3)
    metrics = ['avg_execution_time_per_sample']
    labels = ['Avg. Execution Time (s)']
    cnn_values = [cnn_metrics[m] for m in metrics]
    fusion_values = [fusion_metrics[m] for m in metrics]
    
    x = np.arange(len(labels))
    
    plt.bar(x - width/2, cnn_values, width, label='CNN Only')
    plt.bar(x + width/2, fusion_values, width, label='CNN + SIFT Fusion')
    
    plt.xlabel('Metric')
    plt.ylabel('Time (seconds)')
    plt.title('Execution Time Comparison')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot model size
    plt.subplot(2, 2, 4)
    metrics = ['num_parameters']
    labels = ['Number of Parameters']
    cnn_values = [cnn_metrics[m] / 1e6 for m in metrics]  # Convert to millions
    fusion_values = [fusion_metrics[m] / 1e6 for m in metrics]
    
    x = np.arange(len(labels))
    
    plt.bar(x - width/2, cnn_values, width, label='CNN Only')
    plt.bar(x + width/2, fusion_values, width, label='CNN + SIFT Fusion')
    
    plt.xlabel('Metric')
    plt.ylabel('Parameters (millions)')
    plt.title('Model Size Comparison')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Train and evaluate fusion model for tree species classification')
    parser.add_argument('--data_path', type=str, default="D:\\github\\dataverse_files",
                        help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--view_pooling', type=str, default='max', 
                        choices=['max', 'avg', 'attention'], help='View pooling method')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--save_dir', type=str, default="D:\\github\\tree-species-classification\\models",
                        help='Directory to save the model')
    parser.add_argument('--debug_dir', type=str, default="D:\\github\\tree-species-classification\\debug",
                        help='Directory to save debug information')
    
    args = parser.parse_args()
    
    # Create data loaders
    train_loader, val_loader = create_fusion_data_loaders(
        args.data_path, args.batch_size)
    
    # Create fusion model
    fusion_model = FusionModel(num_classes=len(SPECIES), 
                              pretrained=args.pretrained,
                              view_pooling=args.view_pooling)
    
    # Train fusion model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create debug directory
    os.makedirs(args.debug_dir, exist_ok=True)
    
    # Train the model
    fusion_model, history = train_fusion_model(
        fusion_model, train_loader, val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device,
        debug_dir=args.debug_dir
    )
    
    # Evaluate the model
    print("\nEvaluating fusion model...")
    fusion_metrics = evaluate_fusion_model(fusion_model, val_loader, device)
    
    # Print evaluation report
    print("\nFusion Model Evaluation:")
    print_evaluation_report(fusion_metrics)
    
    # Save model
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(
        args.save_dir, 
        f"fusion_model_{args.view_pooling}_{timestamp}.pth"
    )
    
    torch.save({
        'model_state_dict': fusion_model.state_dict(),
        'args': vars(args),
        'species': SPECIES,
        'history': history,
        'metrics': fusion_metrics
    }, save_path)
    
    print(f"\nModel saved to {save_path}")
    
    # Optional: Compare with CNN-only model
    try:
        # Create and train CNN-only model for comparison
        print("\nTraining CNN-only model for comparison...")
        cnn_model = MultiViewResNet50(
            num_classes=len(SPECIES),
            pretrained=args.pretrained,
            view_pooling=args.view_pooling
        )
        
        cnn_model, _ = fine_tune_model(
            cnn_model, train_loader, val_loader,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            device=device,
            debug_dir=args.debug_dir
        )
        
        # Evaluate CNN-only model
        print("\nEvaluating CNN-only model...")
        cnn_metrics = evaluate_model(cnn_model, val_loader, device)
        
        # Print evaluation report
        print("\nCNN-Only Model Evaluation:")
        print_evaluation_report(cnn_metrics)
        
        # Plot comparison
        print("\nGenerating comparison plot...")
        plot_fusion_comparison(
            cnn_metrics, fusion_metrics,
            save_path=os.path.join(args.debug_dir, f"model_comparison_{timestamp}.png")
        )
    except Exception as e:
        print(f"\nError during comparison: {e}")
        print("Skipping comparison with CNN-only model.")
    
    print("\nDone!")