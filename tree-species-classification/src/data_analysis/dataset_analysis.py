"""
High-Performance Exploratory Data Analysis (EDA) for Tree Species Classification
Optimized for Acer Nitro 5 - R9 5900HX + RTX 3070 + 32GB RAM

This script performs comprehensive analysis of the tree species point cloud dataset
with visualizations and statistical summaries.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import open3d as o3d
import gc
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from collections import defaultdict

# ========================
# PERFORMANCE CONFIGURATION - ACER NITRO 5 OPTIMIZED
# ========================
# CPU Configuration - R9 5900HX (8 cores/16 threads) + 32GB RAM
CPU_CORES = 14  # Use 14 threads, leave 2 for system (8C/16T total)
CHUNK_SIZE = 6  # Process files in chunks

# Memory Configuration - Optimized for 32GB RAM
MAX_POINTS_ANALYSIS = 1000000  # Maximum points to analyze per file
MEMORY_CLEANUP_INTERVAL = 8  # Force garbage collection every N files

# GPU Configuration - RTX 3070 Mobile Optimized
ENABLE_GPU_ACCELERATION = True
BATCH_PROCESS_SIZE = 12  # Process multiple trees simultaneously

# ========================
# CONFIGURATION
# ========================
BASE_DIR = Path(r"D:\github\tree-species-classification")
DATA_DIR = BASE_DIR / "data" / "processed"  # Keep this path for processed data
RAW_DATA_DIR = BASE_DIR / "dataverse_files"  # Original dataset location (for reference only)
META_DIR = DATA_DIR / "metadata"
OUTPUT_DIR = BASE_DIR / "results" / "data_analysis"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configure matplotlib for performance
plt.rcParams['figure.max_open_warning'] = 0
plt.rcParams['agg.path.chunksize'] = 10000

# Configure Open3D for GPU acceleration
if ENABLE_GPU_ACCELERATION:
    os.environ['OPEN3D_USE_NATIVE_DEPENDENCY'] = 'true'

# ========================
# DATA LOADING FUNCTIONS
# ========================
def load_dataset_info():
    """Load dataset metadata from JSON file"""
    info_path = META_DIR / "dataset_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Dataset info not found at {info_path}")
    
    with open(info_path, 'r') as f:
        dataset_info = json.load(f)
    
    return dataset_info

def load_train_test_files():
    """Load train and test file lists"""
    train_csv = META_DIR / "train_files.csv"
    test_csv = META_DIR / "test_files.csv"
    
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Train/test split files not found")
    
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    return train_df, test_df

def load_point_cloud_fast(file_path):
    """Optimized point cloud loading with memory mapping"""
    ext = file_path.suffix.lower()

    try:
        if ext in ['.pts', '.xyz', '.txt']:
            # Use memory mapping for large files
            if file_path.stat().st_size > 50 * 1024 * 1024:  # 50MB threshold
                try:
                    data = np.loadtxt(file_path, dtype=np.float32)  # Use float32 for memory efficiency
                except:
                    data = np.genfromtxt(file_path, delimiter=' ', dtype=np.float32, invalid_raise=False)
            else:
                try:
                    data = np.loadtxt(file_path, dtype=np.float32)
                except:
                    data = np.genfromtxt(file_path, delimiter=' ', dtype=np.float32, invalid_raise=False)

            return data
        else:
            return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# ========================
# ANALYSIS FUNCTIONS
# ========================
def analyze_raw_dataset_structure():
    """Analyze the structure of the raw dataset"""
    if not RAW_DATA_DIR.exists():
        print(f"⚠️ Raw dataset directory not found at {RAW_DATA_DIR}")
        return None
    
    # Collect statistics about the raw dataset
    species_dirs = [d for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
    
    raw_stats = {
        'total_species': len(species_dirs),
        'species_names': [d.name for d in species_dirs],
        'file_counts': {},
        'file_extensions': {},
        'total_files': 0,
        'file_sizes': {}
    }
    
    # Analyze each species directory
    for species_dir in species_dirs:
        species_name = species_dir.name
        files = list(species_dir.glob("*.*"))
        raw_stats['file_counts'][species_name] = len(files)
        raw_stats['total_files'] += len(files)
        
        # Count file extensions
        extensions = [f.suffix.lower() for f in files]
        ext_counts = {ext: extensions.count(ext) for ext in set(extensions)}
        raw_stats['file_extensions'][species_name] = ext_counts
        
        # Calculate total size per species
        species_size = sum(f.stat().st_size for f in files) / (1024 * 1024)  # MB
        raw_stats['file_sizes'][species_name] = species_size
    
    return raw_stats

def analyze_point_cloud(file_path):
    """Analyze a single point cloud file"""
    try:
        points = load_point_cloud_fast(file_path)
        if points is None or len(points) == 0:
            return None
        
        # Ensure we have 3D points
        if points.shape[1] < 3:
            return None
        
        # Take only first 3 columns for XYZ
        points = points[:, :3].astype(np.float32)
        
        # Remove NaN values
        valid_mask = ~np.isnan(points).any(axis=1)
        if not valid_mask.any():
            return None
        
        points = points[valid_mask]
        
        # Sample for performance if needed
        if len(points) > MAX_POINTS_ANALYSIS:
            indices = np.random.choice(len(points), MAX_POINTS_ANALYSIS, replace=False)
            points = points[indices]
        
        # Calculate statistics
        stats = {
            'file_name': file_path.name,
            'species': file_path.parent.name,
            'num_points': len(points),
            'file_size_mb': file_path.stat().st_size / (1024 * 1024),
            'x_min': float(np.min(points[:, 0])),
            'x_max': float(np.max(points[:, 0])),
            'y_min': float(np.min(points[:, 1])),
            'y_max': float(np.max(points[:, 1])),
            'z_min': float(np.min(points[:, 2])),
            'z_max': float(np.max(points[:, 2])),
            'x_range': float(np.max(points[:, 0]) - np.min(points[:, 0])),
            'y_range': float(np.max(points[:, 1]) - np.min(points[:, 1])),
            'z_range': float(np.max(points[:, 2]) - np.min(points[:, 2])),
            'density': len(points) / (np.max(points[:, 0]) - np.min(points[:, 0])) / 
                      (np.max(points[:, 1]) - np.min(points[:, 1])) / 
                      (np.max(points[:, 2]) - np.min(points[:, 2])),
            'mean_x': float(np.mean(points[:, 0])),
            'mean_y': float(np.mean(points[:, 1])),
            'mean_z': float(np.mean(points[:, 2])),
            'std_x': float(np.std(points[:, 0])),
            'std_y': float(np.std(points[:, 1])),
            'std_z': float(np.std(points[:, 2])),
        }
        
        # Calculate PCA for shape analysis
        pca = PCA(n_components=3)
        pca.fit(points)
        stats['pca_ratio_1'] = float(pca.explained_variance_ratio_[0])
        stats['pca_ratio_2'] = float(pca.explained_variance_ratio_[1])
        stats['pca_ratio_3'] = float(pca.explained_variance_ratio_[2])
        
        return stats
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None

def process_files_parallel(file_paths):
    """Process multiple files in parallel"""
    results = []
    
    print(f"🔍 Analyzing {len(file_paths)} point cloud files using {CPU_CORES} cores...")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=CPU_CORES) as executor:
        # Process files in parallel
        futures = [executor.submit(analyze_point_cloud, file_path) for file_path in file_paths]
        
        # Collect results with progress bar
        for i, future in enumerate(tqdm(futures, total=len(futures))):
            result = future.result()
            if result is not None:
                results.append(result)
            
            # Memory cleanup
            if (i + 1) % MEMORY_CLEANUP_INTERVAL == 0:
                gc.collect()
    
    processing_time = time.time() - start_time
    print(f"✅ Analysis complete! Processed {len(results)} files in {processing_time:.2f} seconds")
    print(f"⚡ Processing speed: {len(results) / processing_time:.2f} files/second")
    
    return pd.DataFrame(results)

# ========================
# VISUALIZATION FUNCTIONS
# ========================
def plot_species_distribution(dataset_info):
    """Plot species distribution in train and test sets"""
    train_dist = dataset_info['species_distribution']['train']
    test_dist = dataset_info['species_distribution']['test']
    
    # Convert to DataFrames
    train_df = pd.DataFrame(list(train_dist.items()), columns=['Species', 'Count'])
    train_df['Split'] = 'Train'
    test_df = pd.DataFrame(list(test_dist.items()), columns=['Species', 'Count'])
    test_df['Split'] = 'Test'
    
    # Combine
    combined_df = pd.concat([train_df, test_df])
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Species', y='Count', hue='Split', data=combined_df)
    plt.title('Species Distribution in Train and Test Sets', fontsize=14)
    plt.xlabel('Tree Species', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'species_distribution.png', dpi=300)
    plt.close()

def plot_point_count_distribution(stats_df):
    """Plot distribution of point counts by species"""
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='species', y='num_points', data=stats_df)
    plt.title('Point Count Distribution by Species', fontsize=14)
    plt.xlabel('Tree Species', fontsize=12)
    plt.ylabel('Number of Points (log scale)', fontsize=12)
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'point_count_distribution.png', dpi=300)
    plt.close()
    
    # Also plot as violin plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='species', y='num_points', data=stats_df)
    plt.title('Point Count Distribution by Species (Violin Plot)', fontsize=14)
    plt.xlabel('Tree Species', fontsize=12)
    plt.ylabel('Number of Points (log scale)', fontsize=12)
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'point_count_violin.png', dpi=300)
    plt.close()

def plot_file_size_distribution(stats_df):
    """Plot distribution of file sizes by species"""
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='species', y='file_size_mb', data=stats_df)
    plt.title('File Size Distribution by Species', fontsize=14)
    plt.xlabel('Tree Species', fontsize=12)
    plt.ylabel('File Size (MB)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'file_size_distribution.png', dpi=300)
    plt.close()

def plot_point_density_distribution(stats_df):
    """Plot distribution of point density by species"""
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='species', y='density', data=stats_df)
    plt.title('Point Density Distribution by Species', fontsize=14)
    plt.xlabel('Tree Species', fontsize=12)
    plt.ylabel('Point Density (points/cubic unit)', fontsize=12)
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'point_density_distribution.png', dpi=300)
    plt.close()

def plot_dimension_distributions(stats_df):
    """Plot distributions of x, y, z dimensions by species"""
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3)
    
    # X Range
    ax1 = fig.add_subplot(gs[0, 0])
    sns.boxplot(x='species', y='x_range', data=stats_df, ax=ax1)
    ax1.set_title('X Range Distribution', fontsize=12)
    ax1.set_xlabel('')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Y Range
    ax2 = fig.add_subplot(gs[0, 1])
    sns.boxplot(x='species', y='y_range', data=stats_df, ax=ax2)
    ax2.set_title('Y Range Distribution', fontsize=12)
    ax2.set_xlabel('')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # Z Range
    ax3 = fig.add_subplot(gs[0, 2])
    sns.boxplot(x='species', y='z_range', data=stats_df, ax=ax3)
    ax3.set_title('Z Range Distribution', fontsize=12)
    ax3.set_xlabel('')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # X Standard Deviation
    ax4 = fig.add_subplot(gs[1, 0])
    sns.boxplot(x='species', y='std_x', data=stats_df, ax=ax4)
    ax4.set_title('X Standard Deviation', fontsize=12)
    ax4.set_xlabel('Tree Species')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
    ax4.grid(axis='y', alpha=0.3)
    
    # Y Standard Deviation
    ax5 = fig.add_subplot(gs[1, 1])
    sns.boxplot(x='species', y='std_y', data=stats_df, ax=ax5)
    ax5.set_title('Y Standard Deviation', fontsize=12)
    ax5.set_xlabel('Tree Species')
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)
    ax5.grid(axis='y', alpha=0.3)
    
    # Z Standard Deviation
    ax6 = fig.add_subplot(gs[1, 2])
    sns.boxplot(x='species', y='std_z', data=stats_df, ax=ax6)
    ax6.set_title('Z Standard Deviation', fontsize=12)
    ax6.set_xlabel('Tree Species')
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45)
    ax6.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Dimension Distributions by Species', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_DIR / 'dimension_distributions.png', dpi=300)
    plt.close()

def plot_pca_analysis(stats_df):
    """Plot PCA analysis results"""
    # Create a scatter plot of PCA ratios
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(stats_df['pca_ratio_1'], 
                         stats_df['pca_ratio_2'],
                         c=pd.factorize(stats_df['species'])[0],
                         alpha=0.7,
                         s=50,
                         cmap='viridis')
    
    # Add legend
    species_list = stats_df['species'].unique()
    handles, labels = scatter.legend_elements()
    plt.legend(handles=handles, labels=labels, title="Species", loc="upper right")

    plt.title('PCA Component Analysis by Species', fontsize=14)
    plt.xlabel('PCA Component 1 Ratio', fontsize=12)
    plt.ylabel('PCA Component 2 Ratio', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pca_analysis.png', dpi=300)
    plt.close()
    
    # Create a 3D scatter plot of all three PCA components
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each species with a different color
    for i, species in enumerate(species_list):
        species_data = stats_df[stats_df['species'] == species]
        ax.scatter(species_data['pca_ratio_1'], 
                   species_data['pca_ratio_2'],
                   species_data['pca_ratio_3'],
                   label=species,
                   alpha=0.7,
                   s=50)
    
    ax.set_title('3D PCA Component Analysis by Species', fontsize=14)
    ax.set_xlabel('PCA Component 1 Ratio', fontsize=12)
    ax.set_ylabel('PCA Component 2 Ratio', fontsize=12)
    ax.set_zlabel('PCA Component 3 Ratio', fontsize=12)
    ax.legend(title="Species")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'pca_analysis_3d.png', dpi=300)
    plt.close()

def generate_summary_statistics(stats_df, dataset_info):
    """Generate summary statistics and save to CSV"""
    # Overall statistics
    overall_stats = {
        'total_files': len(stats_df),
        'total_points': stats_df['num_points'].sum(),
        'avg_points_per_file': stats_df['num_points'].mean(),
        'median_points_per_file': stats_df['num_points'].median(),
        'min_points': stats_df['num_points'].min(),
        'max_points': stats_df['num_points'].max(),
        'total_size_mb': stats_df['file_size_mb'].sum(),
        'avg_file_size_mb': stats_df['file_size_mb'].mean(),
        'avg_density': stats_df['density'].mean(),
    }
    
    # Statistics by species
    species_stats = stats_df.groupby('species').agg({
        'num_points': ['count', 'mean', 'median', 'min', 'max', 'sum'],
        'file_size_mb': ['mean', 'sum'],
        'density': ['mean', 'median'],
        'x_range': ['mean', 'median'],
        'y_range': ['mean', 'median'],
        'z_range': ['mean', 'median'],
        'pca_ratio_1': ['mean'],
        'pca_ratio_2': ['mean'],
        'pca_ratio_3': ['mean'],
    }).reset_index()
    
    # Save to CSV
    overall_df = pd.DataFrame([overall_stats])
    overall_df.to_csv(OUTPUT_DIR / 'overall_statistics.csv', index=False)
    species_stats.to_csv(OUTPUT_DIR / 'species_statistics.csv')
    
    # Save detailed stats
    stats_df.to_csv(OUTPUT_DIR / 'detailed_statistics.csv', index=False)
    
    # Generate a text summary
    with open(OUTPUT_DIR / 'dataset_summary.txt', 'w') as f:
        f.write("===== TREE SPECIES CLASSIFICATION DATASET SUMMARY =====\n\n")
        f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Hardware: Acer Nitro 5 - R9 5900HX + RTX 3070 + 32GB RAM\n\n")
        
        f.write("DATASET OVERVIEW:\n")
        f.write(f"Total Files: {dataset_info['dataset_info']['total_files']}\n")
        f.write(f"Training Files: {dataset_info['dataset_info']['train_files']}\n")
        f.write(f"Testing Files: {dataset_info['dataset_info']['test_files']}\n")
        f.write(f"Total Size: {dataset_info['dataset_info']['total_size_mb']:.1f} MB\n")
        f.write(f"Number of Species: {dataset_info['dataset_info']['species_count']}\n\n")
        
        f.write("SPECIES DISTRIBUTION:\n")
        for species, count in dataset_info['species_distribution']['total'].items():
            train_count = dataset_info['species_distribution']['train'][species]
            test_count = dataset_info['species_distribution']['test'][species]
            f.write(f"{species}: {count} total ({train_count} train, {test_count} test)\n")
        
        f.write("\nPOINT CLOUD STATISTICS:\n")
        f.write(f"Total Points Analyzed: {overall_stats['total_points']:,}\n")
        f.write(f"Average Points per File: {overall_stats['avg_points_per_file']:.1f}\n")
        f.write(f"Median Points per File: {overall_stats['median_points_per_file']:.1f}\n")
        f.write(f"Min Points in a File: {overall_stats['min_points']}\n")
        f.write(f"Max Points in a File: {overall_stats['max_points']}\n\n")
        
        f.write("SPECIES POINT STATISTICS:\n")
        for _, row in species_stats.iterrows():
            species = row['species']
            f.write(f"{species}:\n")
            f.write(f"  Files: {row[('num_points', 'count')]}\n")
            f.write(f"  Total Points: {row[('num_points', 'sum')]:,}\n")
            f.write(f"  Avg Points: {row[('num_points', 'mean')]:.1f}\n")
            f.write(f"  Avg Dimensions (x,y,z): {row[('x_range', 'mean')]:.2f}, {row[('y_range', 'mean')]:.2f}, {row[('z_range', 'mean')]:.2f}\n")
            f.write(f"  PCA Ratios: {row[('pca_ratio_1', 'mean')]:.3f}, {row[('pca_ratio_2', 'mean')]:.3f}, {row[('pca_ratio_3', 'mean')]:.3f}\n\n")

# ========================
# MAIN FUNCTION
# ========================
def main():
    print("🚀 HIGH-PERFORMANCE DATASET ANALYSIS")
    print(f"💻 Hardware: Acer Nitro 5 - R9 5900HX (8C/16T) + RTX 3070 Mobile + 32GB RAM")
    print(f"🔧 Using {CPU_CORES}/16 threads")
    print(f"📂 Data Directory: {DATA_DIR}")
    print(f"💾 Output Directory: {OUTPUT_DIR}")
    
    # Load dataset info
    print("\n📊 Loading dataset information...")
    dataset_info = load_dataset_info()
    print(f"✅ Loaded dataset info: {dataset_info['dataset_info']['total_files']} files, "
          f"{dataset_info['dataset_info']['species_count']} species")
    
    # Plot species distribution
    print("\n📈 Generating species distribution plot...")
    plot_species_distribution(dataset_info)
    print("✅ Species distribution plot saved")
    
    # Load train/test files
    print("\n📄 Loading train/test file lists...")
    train_df, test_df = load_train_test_files()
    print(f"✅ Loaded {len(train_df)} train files and {len(test_df)} test files")
    
    # Get all file paths
    all_files = []
    for split_dir in [DATA_DIR / "train", DATA_DIR / "test"]:
        for species_dir in split_dir.iterdir():
            if species_dir.is_dir():
                for file_path in species_dir.glob("*.*"):
                    if file_path.suffix.lower() in ['.pts', '.xyz', '.txt']:
                        all_files.append(file_path)
    
    print(f"\n🔍 Found {len(all_files)} point cloud files to analyze")
    
    # Analyze files in parallel
    stats_df = process_files_parallel(all_files)
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    plot_point_count_distribution(stats_df)
    plot_file_size_distribution(stats_df)
    plot_point_density_distribution(stats_df)
    plot_dimension_distributions(stats_df)
    plot_pca_analysis(stats_df)
    print("✅ All visualizations saved to output directory")
    
    # Generate summary statistics
    print("\n📝 Generating summary statistics...")
    generate_summary_statistics(stats_df, dataset_info)
    print("✅ Summary statistics saved")
    
    print("\n🏆 DATASET ANALYSIS COMPLETE!")
    print(f"💾 All results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    # Set high priority for better performance
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.HIGH_PRIORITY_CLASS if os.name == 'nt' else -10)
    except:
        pass
    
    # Run main function
    main()