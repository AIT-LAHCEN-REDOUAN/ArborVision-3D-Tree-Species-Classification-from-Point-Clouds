"""
High-Performance Tree Species Visualization: Buche (Beech)
Optimized for RTX 3070 & R9 5900HX with multiprocessing and GPU acceleration
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from pathlib import Path
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
from functools import partial
import gc

# ========================
# PERFORMANCE CONFIGURATION - ACER NITRO 5 OPTIMIZED
# ========================
# CPU Configuration - R9 5900HX (8 cores/16 threads) + 32GB RAM
CPU_CORES = 14  # Use 14 threads, leave 2 for system (8C/16T total)
CHUNK_SIZE = 6  # Process files in chunks

# Memory Configuration - Optimized for 32GB RAM
MAX_POINTS_FULL_RENDER = 1000000  # Full quality render threshold (higher with 32GB)
MAX_POINTS_SAMPLED = 200000  # Sampled render for huge datasets
MEMORY_CLEANUP_INTERVAL = 8  # Force garbage collection every N files

# GPU Configuration - RTX 3070 Mobile Optimized
ENABLE_GPU_ACCELERATION = True
BATCH_PROCESS_SIZE = 12  # Process multiple trees simultaneously (more with 32GB)

# ========================
# CONFIGURATION
# ========================
SPECIES_NAME = "Kiefer"
DATA_DIR = Path(r"D:\github\dataverse_files\Kiefer")
OUTPUT_DIR = Path(r"D:/github/tree-species-classification/results/data_visualisations/Kiefer")
OUTPUT_2D = OUTPUT_DIR / "2D"
OUTPUT_3D = OUTPUT_DIR / "3D"

# Create output directories
OUTPUT_2D.mkdir(parents=True, exist_ok=True)
OUTPUT_3D.mkdir(parents=True, exist_ok=True)

# Configure matplotlib for performance
plt.rcParams['figure.max_open_warning'] = 0
plt.rcParams['agg.path.chunksize'] = 10000

# Configure Open3D for GPU acceleration
if ENABLE_GPU_ACCELERATION:
    os.environ['OPEN3D_USE_NATIVE_DEPENDENCY'] = 'true'


# ========================
# OPTIMIZED DATA PROCESSING
# ========================
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
        return None


def preprocess_points_vectorized(points):
    """Vectorized preprocessing for maximum speed"""
    if points is None or points.size == 0:
        return None

    # Ensure we have 3D points
    if points.shape[1] < 3:
        return None

    # Take only first 3 columns for XYZ
    points = points[:, :3].astype(np.float32)

    # Vectorized NaN removal
    valid_mask = ~np.isnan(points).any(axis=1)
    if not valid_mask.any():
        return None

    points = points[valid_mask]

    # Fast duplicate removal using numpy's unique
    if len(points) > 1000:  # Only for larger datasets
        points = np.unique(points, axis=0)

    # Vectorized centering
    if len(points) > 0:
        points -= np.mean(points, axis=0)

    return points


# ========================
# HIGH-PERFORMANCE VISUALIZATION
# ========================
def generate_2d_visualization_fast(points, tree_name, output_path):
    """Optimized 2D visualization with GPU-accelerated rendering"""
    if points is None or len(points) < 100:
        return False

    # Sample for performance if needed
    if len(points) > MAX_POINTS_SAMPLED:
        indices = np.random.choice(len(points), MAX_POINTS_SAMPLED, replace=False)
        points = points[indices]

    # Use subplots with optimized backend
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=100)

    # Pre-calculate colors for all projections
    colors_z = points[:, 2]
    colors_y = points[:, 1]
    colors_x = points[:, 0]

    # Optimized scatter plots with reduced point size and alpha
    s_size = max(0.1, min(1.0, 5000 / len(points)))  # Dynamic point size
    alpha = max(0.3, min(0.8, 10000 / len(points)))  # Dynamic alpha

    # XY projection
    axs[0].scatter(points[:, 0], points[:, 1], s=s_size, alpha=alpha, c=colors_z,
                   cmap='viridis', rasterized=True)
    axs[0].set_title(f'XY - {tree_name}', fontsize=10)
    axs[0].set_xlabel('X', fontsize=8)
    axs[0].set_ylabel('Y', fontsize=8)
    axs[0].grid(True, alpha=0.3)

    # XZ projection
    axs[1].scatter(points[:, 0], points[:, 2], s=s_size, alpha=alpha, c=colors_y,
                   cmap='plasma', rasterized=True)
    axs[1].set_title(f'XZ - {tree_name}', fontsize=10)
    axs[1].set_xlabel('X', fontsize=8)
    axs[1].set_ylabel('Z', fontsize=8)
    axs[1].grid(True, alpha=0.3)

    # YZ projection
    axs[2].scatter(points[:, 1], points[:, 2], s=s_size, alpha=alpha, c=colors_x,
                   cmap='inferno', rasterized=True)
    axs[2].set_title(f'YZ - {tree_name}', fontsize=10)
    axs[2].set_xlabel('Y', fontsize=8)
    axs[2].set_ylabel('Z', fontsize=8)
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout(pad=1.0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

    # Force cleanup
    gc.collect()
    return True


def generate_3d_visualization_gpu_accelerated(points, tree_name, output_path):
    """GPU-accelerated 3D visualization using Open3D with RTX 3070 optimization"""
    if points is None or len(points) < 100:
        return False

    try:
        # Smart sampling for performance
        if len(points) > MAX_POINTS_FULL_RENDER:
            indices = np.random.choice(len(points), MAX_POINTS_FULL_RENDER, replace=False)
            points_render = points[indices]
        else:
            points_render = points

        # Create Open3D point cloud with GPU optimization
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_render.astype(np.float64))

        # Optimized normal estimation (helps with rendering quality)
        if len(points_render) > 1000:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30))

        # Fast colorization by height
        z_coords = np.asarray(pcd.points)[:, 2]
        z_min, z_max = z_coords.min(), z_coords.max()

        if z_max != z_min:
            normalized_z = (z_coords - z_min) / (z_max - z_min)
            colors = plt.cm.viridis(normalized_z)[:, :3]
        else:
            colors = np.tile([0.2, 0.7, 0.3], (len(points_render), 1))

        pcd.colors = o3d.utility.Vector3dVector(colors)

        # High-performance rendering setup
        vis = o3d.visualization.Visualizer()

        # Create window with optimal settings for RTX 3070
        success = vis.create_window(width=1200, height=900, visible=False)
        if not success:
            return False

        vis.add_geometry(pcd)

        # Optimize rendering options
        render_option = vis.get_render_option()
        render_option.background_color = np.array([0.05, 0.05, 0.05])
        render_option.point_size = max(1.0, min(3.0, 100000 / len(points_render)))
        render_option.show_coordinate_frame = False

        # Optimal camera positioning
        ctr = vis.get_view_control()
        if ctr is not None:
            # Set optimal viewing parameters for tree visualization
            ctr.set_zoom(0.7)
            ctr.set_front([0.4, -0.4, -0.8])
            ctr.set_up([0, 0, 1])

            # Center on point cloud
            bbox = pcd.get_axis_aligned_bounding_box()
            ctr.set_lookat(bbox.get_center())

            # Multiple angle renders for best quality
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            # Capture high-quality image
            vis.capture_screen_image(str(output_path))

        vis.destroy_window()
        return True

    except Exception as e:
        return False


def generate_3d_visualization_matplotlib_fast(points, tree_name, output_path):
    """High-performance matplotlib 3D fallback"""
    if points is None or len(points) < 100:
        return False

    # Aggressive sampling for matplotlib performance
    if len(points) > 25000:
        indices = np.random.choice(len(points), 25000, replace=False)
        points_render = points[indices]
    else:
        points_render = points

    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # Fast color calculation
    z_coords = points_render[:, 2]
    z_min, z_max = z_coords.min(), z_coords.max()
    colors = (z_coords - z_min) / (z_max - z_min) if z_max != z_min else np.zeros(len(points_render))

    # Optimized scatter with minimal point size
    point_size = max(0.1, min(2.0, 5000 / len(points_render)))
    ax.scatter(points_render[:, 0], points_render[:, 1], points_render[:, 2],
               c=colors, cmap='viridis', s=point_size, alpha=0.6, rasterized=True)

    ax.set_title(f'3D - {tree_name}', fontsize=12, pad=10)
    ax.set_xlabel('X', fontsize=8)
    ax.set_ylabel('Y', fontsize=8)
    ax.set_zlabel('Z', fontsize=8)

    # Optimize axis limits
    center = np.mean(points_render, axis=0)
    max_range = np.max(np.std(points_render, axis=0)) * 2.5
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)

    # Optimal viewing angle for trees
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

    gc.collect()
    return True


# ========================
# BATCH PROCESSING FUNCTIONS
# ========================
def process_single_tree(args):
    """Process a single tree - designed for multiprocessing"""
    file_path, tree_idx, total_trees = args
    tree_name = file_path.stem

    start_time = time.time()
    result = {
        'tree_name': tree_name,
        'success': False,
        'points_count': 0,
        'processing_time': 0,
        'visualizations': []
    }

    try:
        # Load and preprocess with optimized functions
        points = load_point_cloud_fast(file_path)
        points = preprocess_points_vectorized(points)

        if points is None or len(points) < 100:
            result['error'] = f"Insufficient points: {len(points) if points is not None else 0}"
            return result

        result['points_count'] = len(points)

        # Generate 2D visualization
        output_2d = OUTPUT_2D / f'{tree_name}_2d.png'
        if generate_2d_visualization_fast(points, tree_name, output_2d):
            result['visualizations'].append('2D')

        # Generate 3D visualization with GPU acceleration
        output_3d = OUTPUT_3D / f'{tree_name}_3d.png'
        if generate_3d_visualization_gpu_accelerated(points, tree_name, output_3d):
            result['visualizations'].append('3D_GPU')
        else:
            # Fallback to matplotlib
            output_3d_fallback = OUTPUT_3D / f'{tree_name}_3d_fallback.png'
            if generate_3d_visualization_matplotlib_fast(points, tree_name, output_3d_fallback):
                result['visualizations'].append('3D_CPU')

        result['success'] = True

    except Exception as e:
        result['error'] = str(e)

    result['processing_time'] = time.time() - start_time
    return result


# ========================
# MAIN HIGH-PERFORMANCE PIPELINE
# ========================
def process_species_high_performance():
    """High-performance main processing function with multiprocessing"""
    print(f"🚀 HIGH-PERFORMANCE Processing for {SPECIES_NAME}")
    print(f"💻 Hardware: Acer Nitro 5 - R9 5900HX (8C/16T) + RTX 3070 Mobile + 32GB RAM")
    print(f"🔧 Using {CPU_CORES}/16 threads")
    print(f"📂 Source: {DATA_DIR}")
    print(f"💾 Output: {OUTPUT_DIR}")

    # Get all files
    point_files = list(DATA_DIR.glob('*.*'))
    total_files = len(point_files)
    print(f"🔍 Found {total_files} files")

    if total_files == 0:
        print("❌ No files found!")
        return

    # Prepare arguments for multiprocessing
    process_args = [(file_path, idx, total_files) for idx, file_path in enumerate(point_files)]

    # Performance tracking
    start_time = time.time()
    successful = 0
    failed = 0
    total_points = 0

    print(f"\n🏁 Starting batch processing with {CPU_CORES} processes...")

    # Process in batches optimized for 32GB RAM
    batch_size = min(BATCH_PROCESS_SIZE, CPU_CORES)

    with ProcessPoolExecutor(max_workers=CPU_CORES) as executor:
        # Submit all jobs
        future_to_args = {executor.submit(process_single_tree, args): args
                          for args in process_args}

        # Process results as they complete
        for i, future in enumerate(as_completed(future_to_args)):
            result = future.result()

            if result['success']:
                successful += 1
                total_points += result['points_count']
                viz_types = ', '.join(result['visualizations'])
                print(f"✅ [{successful + failed:3d}/{total_files}] {result['tree_name']} | "
                      f"{result['points_count']:,} pts | {result['processing_time']:.1f}s | {viz_types}")
            else:
                failed += 1
                error_msg = result.get('error', 'Unknown error')
                print(f"❌ [{successful + failed:3d}/{total_files}] {result['tree_name']} | {error_msg}")

            # Memory cleanup
            if (i + 1) % MEMORY_CLEANUP_INTERVAL == 0:
                gc.collect()

    # Final statistics
    total_time = time.time() - start_time
    avg_time_per_tree = total_time / total_files
    avg_points_per_tree = total_points / successful if successful > 0 else 0
    points_per_second = total_points / total_time if total_time > 0 else 0

    print(f"\n🏆 HIGH-PERFORMANCE PROCESSING COMPLETE!")
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"📊 Success rate: {successful}/{total_files} ({100 * successful / total_files:.1f}%)")
    print(f"⚡ Avg time per tree: {avg_time_per_tree:.2f}s")
    print(f"📈 Avg points per tree: {avg_points_per_tree:,.0f}")
    print(f"🚄 Processing speed: {points_per_second:,.0f} points/second")
    print(f"💾 Output saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    # Set high priority for better performance
    try:
        import psutil

        p = psutil.Process()
        p.nice(psutil.HIGH_PRIORITY_CLASS if os.name == 'nt' else -10)
    except:
        pass

    # Suppress warnings for cleaner output
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        process_species_high_performance()