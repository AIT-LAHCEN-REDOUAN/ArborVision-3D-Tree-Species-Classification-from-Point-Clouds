# Tree Species Classification

This repository contains code for classifying tree species from 3D point cloud data using various approaches. The implementation is optimized for an Acer Nitro 5 laptop with R9 5900HX CPU, RTX 3070 GPU, and 32GB RAM.

## Project Structure

```
tree-species-classification/
├── data/                      # Data directory
│   ├── raw/                   # Raw point cloud data
│   └── processed/             # Processed data
│       ├── train/             # Training data split by species
│       ├── test/              # Test data split by species
│       └── multiview_images/  # Generated multi-view images
├── models/                    # Saved models
├── results/                   # Results and visualizations
├── src/                       # Source code
│   └── Classification_methods/
│       └── Indirect_Methods/
│           └── Multi-view_Classical_Descriptors_ML/
│               ├── multi_view_generator.py        # Multi-view image generation
│               └── multi_view_classical_ml_pipeline.py  # ML pipeline
└── README.md                  # This file
```

## Data Preparation

Before running the classification pipeline, you need to prepare the data by splitting it into train and test sets:

```bash
python src/data_splitting.py
```

This script will:
1. Discover all point cloud files (`.xyz`, `.pts`, `.txt`) in the source directory
2. Split the data into train and test sets based on the provided test.csv file
3. Copy files to the appropriate directories
4. Generate metadata about the dataset

## Multi-view Classification Pipeline

The multi-view classification approach consists of two main steps:

1. Generate multi-view images from 3D point clouds
2. Extract classical image descriptors and train ML models

### 1. Multi-view Image Generation

The `multi_view_generator.py` script converts 3D point cloud data into 2D multi-view projections:

```bash
python src/Classification_methods/Indirect_Methods/Multi-view_Classical_Descriptors_ML/multi_view_generator.py --num_views 8 --resolution 224 --split train
```

Options:
- `--num_views`: Number of views to generate per point cloud (default: 8)
- `--resolution`: Resolution of generated images (default: 224)
- `--split`: Which data split to process ('train', 'test', or 'both') (default: 'both')

This will generate multi-view images for each point cloud and save them in the `data/processed/multiview_images/{split}/{species}/` directories.

### 2. Classical ML Pipeline

The `multi_view_classical_ml_pipeline.py` script extracts features from the multi-view images and trains classical ML models:

```bash
python src/Classification_methods/Indirect_Methods/Multi-view_Classical_Descriptors_ML/multi_view_classical_ml_pipeline.py --features hog lbp sift color --combination mean
```

Options:
- `--features`: Feature types to extract (hog, lbp, sift, color) (default: all)
- `--combination`: Method to combine multi-view features ('mean', 'max', 'concat') (default: 'mean')

The script will:
1. Load the multi-view images
2. Extract the specified features from each image
3. Combine features from multiple views of the same tree
4. Train and evaluate multiple ML models (RandomForest, SVM, GradientBoosting)
5. Save the trained models and evaluation results

## Expected Data Structure

The input point cloud data should be organized by species:

```
dataverse_files/
├── Buche/
│   ├── tree1.xyz
│   ├── tree2.pts
│   └── ...
├── Douglasie/
│   ├── tree1.xyz
│   └── ...
└── ...
```

After running the data splitting script, the processed data will be organized as:

```
data/processed/
├── train/
│   ├── Buche/
│   │   ├── tree1.xyz
│   │   └── ...
│   ├── Douglasie/
│   │   ├── tree1.xyz
│   │   └── ...
│   └── ...
└── test/
    ├── Buche/
    │   ├── tree2.pts
    │   └── ...
    └── ...
```

After running the multi-view generator, the images will be organized as:

```
data/processed/multiview_images/
├── train/
│   ├── Buche/
│   │   ├── tree1_view0.png
│   │   ├── tree1_view1.png
│   │   └── ...
│   └── ...
└── test/
    ├── Buche/
    │   ├── tree2_view0.png
    │   └── ...
    └── ...
```

## Requirements

See `requirements.txt` for the full list of dependencies. The main requirements are:

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- OpenCV
- Open3D
- PyTorch (for GPU acceleration)
- CuPy (for CUDA acceleration)
- PyOpenCL (optional, for OpenCL acceleration)

## Hardware Optimization

The code is optimized for an Acer Nitro 5 laptop with:
- AMD Ryzen 9 5900HX CPU
- NVIDIA RTX 3070 GPU
- 32GB RAM

Hardware acceleration is used when available, with fallbacks to CPU processing when necessary.