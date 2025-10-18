import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

SPECIES = ["Buche","Douglasie","Eiche","Esche","Fichte","Kiefer","Roteiche"]

# Default dataset path
DEFAULT_DATASET_PATH = "D:\\github\\dataverse_files"
# Default output path for rendered views
DEFAULT_OUTPUT_PATH = "D:\\github\\tree-species-classification\\results\\multi_view_images"

def read_points(path):
    """Load point cloud from txt/pts/xyz file."""
    pts = []
    with open(path,"r") as f:
        for ln in f:
            p = ln.strip().split()
            if len(p) >= 3:
                try:
                    pts.append([float(p[0]), float(p[1]), float(p[2])])
                except:
                    pass
    a = np.asarray(pts, dtype=np.float32)
    if a.size == 0:
        a = np.zeros((1,3), np.float32)
    return a

def normalize_points(P):
    """Center and scale point cloud to unit sphere."""
    c = P.mean(0, keepdims=True)
    P = P - c
    s = np.max(np.linalg.norm(P, axis=1)) or 1.0
    return P / s

def render_views(P, size=224, padding=2, num_azimuth=12, num_elev=3, save_path=None, filename=None):
    """
    Render orthographic projections by rotating around azimuth and elevation.
    Returns array of shape [V, H, W] (grayscale).
    
    If save_path and filename are provided, saves the rendered views to disk.
    """
    from scipy.ndimage import gaussian_filter

    P = normalize_points(P)
    chans = []

    # Elevation angles: e.g. top, middle, bottom
    elev_angles = np.linspace(-30, 30, num_elev)  # in degrees
    
    # Create directory if saving is enabled
    if save_path and filename:
        os.makedirs(save_path, exist_ok=True)
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        species_dir = None
        
        # Try to extract species from filepath
        for species in SPECIES:
            if species in filename:
                species_dir = os.path.join(save_path, species)
                os.makedirs(species_dir, exist_ok=True)
                break
                
        if not species_dir:
            species_dir = os.path.join(save_path, "unknown")
            os.makedirs(species_dir, exist_ok=True)

    view_idx = 0
    for phi_idx, phi in enumerate(elev_angles):           # elevation tilt
        phi = np.deg2rad(phi)
        Rphi = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi),  np.cos(phi)]
        ], dtype=np.float32)

        for v in range(num_azimuth):  # azimuth rotation
            theta = 2 * np.pi * v / num_azimuth
            Rtheta = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0,              0,             1]
            ], dtype=np.float32)

            R = Rtheta @ Rphi
            Prot = P @ R.T
            pts = Prot[:, [0,1]]
            pts = (pts + 1.0) / 2.0
            img = np.zeros((size, size), dtype=np.float32)
            idx = (pts * (size - 1 - padding*2) + padding).clip(0, size-1).astype(int)
            img[idx[:,1], idx[:,0]] = 1.0
            img = gaussian_filter(img, sigma=1.0)
            img = (img / (img.max() or 1.0)) * 255.0
            img_uint8 = img.astype(np.uint8)
            chans.append(img_uint8)
            
            # Save the image if path is provided
            if save_path and filename:
                img_pil = Image.fromarray(img_uint8)
                img_path = os.path.join(species_dir, f"{base_filename}_elev{phi_idx}_azim{v}.png")
                img_pil.save(img_path)
                view_idx += 1

    arr = np.stack(chans, axis=0)  # [V,H,W], V = num_azimuth*num_elev
    return arr


class MultiViewPCDataset(Dataset):
    def __init__(self, root=DEFAULT_DATASET_PATH, test_list=None, size=224, cache=True, 
                 num_azimuth=12, num_elev=3, save_views=False, 
                 output_path=DEFAULT_OUTPUT_PATH):
        """
        root: root directory with species subfolders
        test_list: optional set of filenames reserved for test (to exclude)
        save_views: if True, save rendered views to output_path
        output_path: directory to save rendered views
        """
        self.samples = []
        for ci, cls in enumerate(SPECIES):
            for ext in ("*.pts","*.txt","*.xyz"):
                for fp in glob.glob(os.path.join(root, cls, ext)):
                    fname = os.path.basename(fp)
                    if test_list and fname in test_list:
                        continue  # exclude test samples
                    self.samples.append((fp, ci))
        self.size = size
        self.cache = cache
        self.num_azimuth = num_azimuth
        self.num_elev = num_elev
        self._cache = {}
        self.save_views = save_views
        self.output_path = output_path
        
        # Create output directory if saving is enabled
        if self.save_views:
            os.makedirs(self.output_path, exist_ok=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fp, label = self.samples[idx]
        if self.cache and fp in self._cache:
            X = self._cache[fp]
        else:
            P = read_points(fp)
            save_path = self.output_path if self.save_views else None
            X = render_views(P, self.size, num_azimuth=self.num_azimuth, 
                            num_elev=self.num_elev, save_path=save_path, 
                            filename=fp)  # [V,H,W]
            if self.cache:
                self._cache[fp] = X
        # [V,1,H,W] float tensor in [0,1]
        X = torch.from_numpy(X.astype(np.float32)/255.0).unsqueeze(1)
        return X, label, fp


# Add this at the end of the file

if __name__ == "__main__":
    # Create dataset with save_views=True to save all rendered views
    dataset = MultiViewPCDataset(save_views=True)
    
    print(f"Processing {len(dataset)} point cloud files...")
    print(f"Saving rendered views to: {dataset.output_path}")
    
    # Process all samples to generate and save the views
    for i in range(len(dataset)):
        # Get the sample but don't need to use the returned tensors
        _, _, filepath = dataset[i]
        print(f"Processed {i+1}/{len(dataset)}: {os.path.basename(filepath)}")
    
    print(f"\nAll views have been saved to: {dataset.output_path}")
    print(f"Total files processed: {len(dataset)}")
