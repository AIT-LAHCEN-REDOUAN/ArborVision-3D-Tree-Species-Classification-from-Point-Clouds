"""
High-Performance Data Splitting for Tree Species Classification
Optimized for Acer Nitro 5 - R9 5900HX + RTX 3070 + 32GB RAM

Creates train/test split based on test.csv and organizes data for ML pipeline
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from collections import Counter
import json
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from tqdm import tqdm

# ========================
# CONFIGURATION
# ========================
# Base directories
BASE_DIR = Path(r"D:\github\tree-species-classification")
DATA_DIR = Path(r"D:\github\dataverse_files")
OUTPUT_DIR = BASE_DIR / "data" / "processed"
TEST_CSV_PATH = DATA_DIR / "test.csv"  # Your test.csv file

# Performance settings for your hardware
NUM_THREADS = 12  # Optimized for 8C/16T CPU
CHUNK_SIZE = 50  # Process files in chunks for better memory management

# Output structure
TRAIN_DIR = OUTPUT_DIR / "train"
TEST_DIR = OUTPUT_DIR / "test"
METADATA_DIR = OUTPUT_DIR / "metadata"

# Create directories
for dir_path in [TRAIN_DIR, TEST_DIR, METADATA_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ========================
# DATA ANALYSIS FUNCTIONS
# ========================
def load_test_csv(csv_path):
    """Load and analyze test.csv file"""
    print(f"📄 Loading test.csv from: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"test.csv not found at {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} test samples")

    # Clean and standardize paths
    df['path'] = df['path'].str.replace('\\', '/')
    # Fix character encoding issues (replace Cyrillic 'Б' with German 'ü')
    df['path'] = df['path'].str.replace('Б', 'ü')
    df['species'] = df['class']  # Rename for clarity

    return df


def discover_all_files(data_dir):
    """Discover all point cloud files in the dataset"""
    print(f"🔍 Discovering all files in: {data_dir}")

    # Supported file extensions
    extensions = ['*.xyz', '*.pts', '*.txt']
    all_files = []

    # Find all species directories
    species_dirs = [d for d in data_dir.iterdir() if d.is_dir()]

    for species_dir in species_dirs:
        species_name = species_dir.name
        print(f"  📂 Scanning {species_name}...")

        species_files = []
        for ext in extensions:
            species_files.extend(species_dir.glob(ext))

        # Create relative paths for matching with test.csv
        for file_path in species_files:
            relative_path = f"{species_name}/{file_path.name}"
            all_files.append({
                'relative_path': relative_path,
                'absolute_path': file_path,
                'species': species_name,
                'file_size': file_path.stat().st_size if file_path.exists() else 0
            })

        print(f"    Found {len(species_files)} files")

    return pd.DataFrame(all_files)


def analyze_dataset_distribution(all_files_df, test_df):
    """Analyze the distribution of species in train/test splits"""
    print(f"\n📊 DATASET ANALYSIS")
    print(f"=" * 50)

    # Test set analysis
    test_species_count = test_df['species'].value_counts()
    print(f"🧪 TEST SET ({len(test_df)} samples):")
    for species, count in test_species_count.items():
        percentage = (count / len(test_df)) * 100
        print(f"   {species:12}: {count:3d} samples ({percentage:5.1f}%)")

    # Full dataset analysis
    all_species_count = all_files_df['species'].value_counts()
    print(f"\n🗄  FULL DATASET ({len(all_files_df)} samples):")
    for species, count in all_species_count.items():
        percentage = (count / len(all_files_df)) * 100
        test_count = test_species_count.get(species, 0)
        train_count = count - test_count
        train_percentage = (train_count / count) * 100 if count > 0 else 0
        print(
            f"   {species:12}: {count:3d} total | {train_count:3d} train ({train_percentage:5.1f}%) | {test_count:3d} test")

    return test_species_count, all_species_count


def copy_file_safe(args):
    """Safely copy a file with error handling"""
    src_path, dst_path = args
    try:
        # Create destination directory if it doesn't exist
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file
        shutil.copy2(src_path, dst_path)
        return {
            'success': True,
            'src': str(src_path),
            'dst': str(dst_path),
            'size': src_path.stat().st_size
        }
    except Exception as e:
        return {
            'success': False,
            'src': str(src_path),
            'dst': str(dst_path),
            'error': str(e)
        }


# ========================
# MAIN SPLITTING FUNCTIONS
# ========================
def create_data_splits(test_csv_path, data_dir):
    """Main function to create train/test splits"""
    start_time = time.time()

    print(f"🚀 HIGH-PERFORMANCE DATA SPLITTING")
    print(f"💻 Hardware: Acer Nitro 5 - R9 5900HX + RTX 3070 + 32GB RAM")
    print(f"🔧 Using {NUM_THREADS} threads")
    print(f"=" * 60)

    # Step 1: Load test.csv
    test_df = load_test_csv(test_csv_path)

    # Step 2: Discover all files
    all_files_df = discover_all_files(data_dir)

    # Step 3: Analyze distributions
    test_species_count, all_species_count = analyze_dataset_distribution(all_files_df, test_df)

    # Step 4: Create file mapping
    print(f"\n🔗 CREATING FILE MAPPINGS...")

    # Convert test.csv paths to set for fast lookup
    test_paths = set(test_df['path'].tolist())

    # Split files into train and test
    test_files = []
    train_files = []
    missing_files = []

    for _, row in all_files_df.iterrows():
        if row['relative_path'] in test_paths:
            test_files.append(row)
        else:
            train_files.append(row)

    # Check for missing test files
    found_test_paths = set([f['relative_path'] for f in test_files])
    missing_test_paths = test_paths - found_test_paths

    if missing_test_paths:
        print(f"⚠  WARNING: {len(missing_test_paths)} test files not found:")
        for path in list(missing_test_paths)[:10]:  # Show first 10
            print(f"     {path}")
        if len(missing_test_paths) > 10:
            print(f"     ... and {len(missing_test_paths) - 10} more")

    print(f"✅ File mapping complete:")
    print(f"   📚 Training files: {len(train_files)}")
    print(f"   🧪 Test files: {len(test_files)}")
    print(f"   ❌ Missing files: {len(missing_test_paths)}")

    # Step 5: Copy files with multiprocessing
    print(f"\n📁 COPYING FILES...")

    # Prepare copy tasks
    copy_tasks = []

    # Test files
    for file_info in test_files:
        src_path = file_info['absolute_path']
        dst_path = TEST_DIR / file_info['species'] / file_info['absolute_path'].name
        copy_tasks.append((src_path, dst_path))

    # Train files
    for file_info in train_files:
        src_path = file_info['absolute_path']
        dst_path = TRAIN_DIR / file_info['species'] / file_info['absolute_path'].name
        copy_tasks.append((src_path, dst_path))

    print(f"🔄 Processing {len(copy_tasks)} files with {NUM_THREADS} threads...")

    # Execute file copying with progress bar
    successful_copies = 0
    failed_copies = 0
    total_size = 0

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(copy_file_safe, task): task for task in copy_tasks}

        # Process results with progress bar
        with tqdm(total=len(copy_tasks), desc="Copying files") as pbar:
            for future in future_to_task:
                result = future.result()

                if result['success']:
                    successful_copies += 1
                    total_size += result.get('size', 0)
                else:
                    failed_copies += 1
                    print(f"\n❌ Failed to copy {result['src']}: {result['error']}")

                pbar.update(1)

    # Step 6: Generate metadata
    print(f"\n📋 GENERATING METADATA...")

    metadata = {
        'dataset_info': {
            'creation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_files': len(all_files_df),
            'train_files': len(train_files),
            'test_files': len(test_files),
            'missing_files': len(missing_test_paths),
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'species_count': len(all_species_count)
        },
        'species_distribution': {
            'train': {},
            'test': {},
            'total': {}
        },
        'file_paths': {
            'train_dir': str(TRAIN_DIR),
            'test_dir': str(TEST_DIR),
            'source_dir': str(data_dir)
        }
    }

    # Calculate species distributions
    train_df = pd.DataFrame(train_files)
    test_df_found = pd.DataFrame(test_files)

    if not train_df.empty:
        train_species = train_df['species'].value_counts().to_dict()
        metadata['species_distribution']['train'] = train_species

    if not test_df_found.empty:
        test_species = test_df_found['species'].value_counts().to_dict()
        metadata['species_distribution']['test'] = test_species

    metadata['species_distribution']['total'] = all_species_count.to_dict()

    # Save metadata
    metadata_file = METADATA_DIR / 'dataset_info.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save file lists
    train_list_file = METADATA_DIR / 'train_files.csv'
    test_list_file = METADATA_DIR / 'test_files.csv'

    if train_files:
        pd.DataFrame(train_files).to_csv(train_list_file, index=False)

    if test_files:
        pd.DataFrame(test_files).to_csv(test_list_file, index=False)

    # Final summary
    processing_time = time.time() - start_time

    print(f"\n🏆 DATA SPLITTING COMPLETE!")
    print(f"=" * 60)
    print(f"⏱  Total time: {processing_time:.1f}s")
    print(f"📊 Success rate: {successful_copies}/{len(copy_tasks)} ({100 * successful_copies / len(copy_tasks):.1f}%)")
    print(f"💾 Data size: {metadata['dataset_info']['total_size_mb']:.1f} MB")
    print(f"📁 Output directories:")
    print(f"   🚆 Training: {TRAIN_DIR} ({len(train_files)} files)")
    print(f"   🧪 Testing:  {TEST_DIR} ({len(test_files)} files)")
    print(f"   📋 Metadata: {METADATA_DIR}")

    print(f"\n📈 SPECIES DISTRIBUTION SUMMARY:")
    for species in sorted(all_species_count.keys()):
        total = all_species_count[species]
        train_count = metadata['species_distribution']['train'].get(species, 0)
        test_count = metadata['species_distribution']['test'].get(species, 0)
        train_pct = (train_count / total * 100) if total > 0 else 0
        print(
            f"   {species:12}: {total:3d} total ({train_count:3d} train, {test_count:2d} test) - {train_pct:5.1f}% train")

    return metadata


# ========================
# VALIDATION FUNCTIONS
# ========================
def validate_splits():
    """Validate the created train/test splits"""
    print(f"\n🔍 VALIDATING SPLITS...")

    validation_results = {
        'train_valid': True,
        'test_valid': True,
        'issues': []
    }

    # Check if directories exist and contain files
    if not TRAIN_DIR.exists() or not any(TRAIN_DIR.iterdir()):
        validation_results['train_valid'] = False
        validation_results['issues'].append("Training directory empty or missing")

    if not TEST_DIR.exists() or not any(TEST_DIR.iterdir()):
        validation_results['test_valid'] = False
        validation_results['issues'].append("Test directory empty or missing")

    # Count files in each split
    train_files = []
    test_files = []

    if TRAIN_DIR.exists():
        for species_dir in TRAIN_DIR.iterdir():
            if species_dir.is_dir():
                species_files = list(species_dir.glob('*'))
                train_files.extend(species_files)

    if TEST_DIR.exists():
        for species_dir in TEST_DIR.iterdir():
            if species_dir.is_dir():
                species_files = list(species_dir.glob('*'))
                test_files.extend(species_files)

    print(f"✅ Validation complete:")
    print(f"   📚 Training files found: {len(train_files)}")
    print(f"   🧪 Test files found: {len(test_files)}")
    print(
        f"   ⚡ Status: {'✅ VALID' if validation_results['train_valid'] and validation_results['test_valid'] else '❌ ISSUES FOUND'}")

    if validation_results['issues']:
        print(f"   ⚠  Issues:")
        for issue in validation_results['issues']:
            print(f"      - {issue}")

    return validation_results


# ========================
# MAIN EXECUTION
# ========================
if __name__ == "__main__":
    try:
        # Set high priority for better performance
        try:
            import psutil

            p = psutil.Process()
            p.nice(psutil.HIGH_PRIORITY_CLASS if os.name == 'nt' else -10)
        except:
            pass

        # Execute data splitting
        metadata = create_data_splits(TEST_CSV_PATH, DATA_DIR)

        # Validate results
        validation_results = validate_splits()

        if validation_results['train_valid'] and validation_results['test_valid']:
            print(f"\n🎉 SUCCESS! Dataset ready for machine learning pipeline!")
            print(f"📖 Next steps:")
            print(f"   1. Load training data from: {TRAIN_DIR}")
            print(f"   2. Load test data from: {TEST_DIR}")
            print(f"   3. Check metadata in: {METADATA_DIR}")
        else:
            print(f"\n⚠  WARNING: Issues found during validation. Please check the output above.")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback

        traceback.print_exc()