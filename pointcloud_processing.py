import numpy as np
import open3d as o3d
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PointCloudProcessor:
    def __init__(self, ply_path):
        self.ply_path = Path(ply_path)
        self.point_cloud = None
        self.load_point_cloud()
    
    def load_point_cloud(self):
        """Load point cloud from PLY file"""
        if not self.ply_path.exists():
            print(f"Point cloud file not found: {self.ply_path}")
            return False
        
        try:
            self.point_cloud = o3d.io.read_point_cloud(str(self.ply_path))
            print(f"Loaded point cloud with {len(self.point_cloud.points)} points")
            return True
        except Exception as e:
            print(f"Error loading point cloud: {e}")
            return False
    
    def get_point_cloud_info(self):
        """Get basic information about the point cloud"""
        if self.point_cloud is None:
            return None
        
        points = np.asarray(self.point_cloud.points)
        colors = np.asarray(self.point_cloud.colors)
        
        info = {
            'num_points': len(points),
            'has_colors': len(colors) > 0,
            'bounds': {
                'min': points.min(axis=0),
                'max': points.max(axis=0),
                'center': points.mean(axis=0)
            }
        }
        
        return info
    
    def filter_outliers(self, nb_neighbors=20, std_ratio=2.0):
        """Remove outliers from point cloud"""
        if self.point_cloud is None:
            return False
        
        print("Removing outliers...")
        cl, ind = self.point_cloud.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        
        outlier_cloud = self.point_cloud.select_by_index(ind, invert=True)
        self.point_cloud = self.point_cloud.select_by_index(ind)
        
        print(f"Removed {len(outlier_cloud.points)} outlier points")
        return True
    
    def downsample(self, voxel_size=0.05):
        """Downsample point cloud to reduce density"""
        if self.point_cloud is None:
            return False
        
        print(f"Downsampling with voxel size: {voxel_size}")
        original_size = len(self.point_cloud.points)
        
        self.point_cloud = self.point_cloud.voxel_down_sample(voxel_size)
        
        print(f"Downsampled from {original_size} to {len(self.point_cloud.points)} points")
        return True
    
    def estimate_normals(self):
        """Estimate normals for the point cloud"""
        if self.point_cloud is None:
            return False
        
        print("Estimating normals...")
        self.point_cloud.estimate_normals()
        self.point_cloud.orient_normals_consistent_tangent_plane(100)
        return True
    
    def create_mesh(self, method='poisson'):
        """Create mesh from point cloud"""
        if self.point_cloud is None:
            return None
        
        print(f"Creating mesh using {method} method...")
        
        if method == 'poisson':
            # Poisson reconstruction
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                self.point_cloud, depth=9)
        elif method == 'alpha':
            # Alpha shapes
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                self.point_cloud, 0.03)
        else:
            print(f"Unknown mesh method: {method}")
            return None
        
        # Remove degenerate triangles
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        print(f"Created mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
        return mesh
    
    def visualize(self, show_mesh=False):
        """Visualize the point cloud"""
        if self.point_cloud is None:
            print("No point cloud to visualize")
            return
        
        print("Launching 3D visualization...")
        
        geometries = [self.point_cloud]
        
        if show_mesh:
            mesh = self.create_mesh()
            if mesh is not None:
                geometries.append(mesh)
        
        # Create coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        geometries.append(coord_frame)
        
        o3d.visualization.draw_geometries(geometries)
    
    def save_processed_cloud(self, output_path):
        """Save processed point cloud"""
        if self.point_cloud is None:
            return False
        
        try:
            o3d.io.write_point_cloud(str(output_path), self.point_cloud)
            print(f"Saved processed point cloud to: {output_path}")
            return True
        except Exception as e:
            print(f"Error saving point cloud: {e}")
            return False
    
    def export_to_other_formats(self, output_dir):
        """Export point cloud to various formats"""
        if self.point_cloud is None:
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Export to PLY
        ply_path = output_dir / "processed_cloud.ply"
        o3d.io.write_point_cloud(str(ply_path), self.point_cloud)
        
        # Export to PCD
        pcd_path = output_dir / "processed_cloud.pcd"
        o3d.io.write_point_cloud(str(pcd_path), self.point_cloud)
        
        # Export to XYZ
        xyz_path = output_dir / "processed_cloud.xyz"
        o3d.io.write_point_cloud(str(xyz_path), self.point_cloud)
        
        # Create mesh and export
        mesh = self.create_mesh()
        if mesh is not None:
            mesh_path = output_dir / "mesh.obj"
            o3d.io.write_triangle_mesh(str(mesh_path), mesh)
            
            # Export mesh as STL
            stl_path = output_dir / "mesh.stl"
            o3d.io.write_triangle_mesh(str(stl_path), mesh)
        
        print(f"Exported point cloud and mesh to: {output_dir}")

class PointCloudAnalyzer:
    def __init__(self, point_cloud):
        self.point_cloud = point_cloud
    
    def analyze_density(self):
        """Analyze point cloud density"""
        if self.point_cloud is None:
            return None
        
        points = np.asarray(self.point_cloud.points)
        
        # Calculate distances to nearest neighbors
        pcd_tree = o3d.geometry.KDTreeFlann(self.point_cloud)
        distances = []
        
        for i in range(len(points)):
            [k, idx, _] = pcd_tree.search_knn_vector_3d(points[i], 2)
            if k >= 2:
                dist = np.linalg.norm(points[i] - points[idx[1]])
                distances.append(dist)
        
        distances = np.array(distances)
        
        density_info = {
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances)
        }
        
        return density_info
    
    def plot_density_histogram(self):
        """Plot histogram of point densities"""
        density_info = self.analyze_density()
        if density_info is None:
            return
        
        points = np.asarray(self.point_cloud.points)
        pcd_tree = o3d.geometry.KDTreeFlann(self.point_cloud)
        distances = []
        
        for i in range(len(points)):
            [k, idx, _] = pcd_tree.search_knn_vector_3d(points[i], 2)
            if k >= 2:
                dist = np.linalg.norm(points[i] - points[idx[1]])
                distances.append(dist)
        
        plt.figure(figsize=(10, 6))
        plt.hist(distances, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Distance to Nearest Neighbor')
        plt.ylabel('Frequency')
        plt.title('Point Cloud Density Distribution')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def create_height_map(self, resolution=0.1):
        """Create a height map from the point cloud"""
        if self.point_cloud is None:
            return None
        
        points = np.asarray(self.point_cloud.points)
        
        # Project points to XY plane
        x_min, y_min = points[:, :2].min(axis=0)
        x_max, y_max = points[:, :2].max(axis=0)
        
        # Create grid
        x_bins = int((x_max - x_min) / resolution) + 1
        y_bins = int((y_max - y_min) / resolution) + 1
        
        height_map = np.full((y_bins, x_bins), np.nan)
        
        # Fill height map
        for point in points:
            x_idx = int((point[0] - x_min) / resolution)
            y_idx = int((point[1] - y_min) / resolution)
            
            if 0 <= x_idx < x_bins and 0 <= y_idx < y_bins:
                if np.isnan(height_map[y_idx, x_idx]):
                    height_map[y_idx, x_idx] = point[2]
                else:
                    height_map[y_idx, x_idx] = max(height_map[y_idx, x_idx], point[2])
        
        return height_map

def create_summary_report(colmap_project_path):
    """Create a summary report of the reconstruction"""
    project_path = Path(colmap_project_path)
    
    print("=== COLMAP Reconstruction Summary ===")
    
    # Check sparse reconstruction
    sparse_path = project_path / "sparse" / "0"
    if sparse_path.exists():
        print(f"✓ Sparse reconstruction found at: {sparse_path}")
        
        # Read cameras, images, and points3D files
        cameras_file = sparse_path / "cameras.txt"
        images_file = sparse_path / "images.txt"
        points3d_file = sparse_path / "points3D.txt"
        
        if cameras_file.exists():
            with open(cameras_file, 'r') as f:
                cameras = [line for line in f if not line.startswith('#')]
            print(f"  - Cameras calibrated: {len(cameras)}")
        
        if images_file.exists():
            with open(images_file, 'r') as f:
                images = [line for line in f if not line.startswith('#')]
            print(f"  - Images registered: {len(images)}")
        
        if points3d_file.exists():
            with open(points3d_file, 'r') as f:
                points = [line for line in f if not line.startswith('#')]
            print(f"  - 3D points: {len(points)}")
    
    # Check dense reconstruction
    dense_ply = project_path / "dense" / "fused.ply"
    if dense_ply.exists():
        print(f"✓ Dense point cloud found at: {dense_ply}")
        
        # Load and analyze dense point cloud
        try:
            pcd = o3d.io.read_point_cloud(str(dense_ply))
            print(f"  - Dense points: {len(pcd.points)}")
            
            points = np.asarray(pcd.points)
            bounds = {
                'min': points.min(axis=0),
                'max': points.max(axis=0),
                'size': points.max(axis=0) - points.min(axis=0)
            }
            print(f"  - Bounds: {bounds['size']}")
            
            if len(pcd.colors) > 0:
                print("  - Point cloud has colors")
            
        except Exception as e:
            print(f"  - Error reading dense point cloud: {e}")
    
    print("\n=== Next Steps ===")
    print("1. Use the PointCloudProcessor to clean and process the point cloud")
    print("2. Visualize the results with the visualization functions")
    print("3. Export to desired formats (PLY, OBJ, STL, etc.)")

# Usage example and main execution
if __name__ == "__main__":
    # Example usage
    colmap_project = r"D:\Multi_Camera_Tracking\Wildtrack\colmap_input"
    
    # Create summary report
    create_summary_report(colmap_project)
    
    # Process point cloud if it exists
    dense_ply = Path(colmap_project) / "dense" / "fused.ply"
    
    if dense_ply.exists():
        print("\nProcessing point cloud...")
        
        # Initialize processor
        processor = PointCloudProcessor(dense_ply)
        
        # Get info about the point cloud
        info = processor.get_point_cloud_info()
        if info:
            print(f"Point cloud info: {info}")
        
        # Process the point cloud
        processor.filter_outliers(nb_neighbors=20, std_ratio=2.0)
        processor.downsample(voxel_size=0.02)  # Adjust voxel size as needed
        processor.estimate_normals()
        
        # Save processed cloud
        output_dir = Path(colmap_project) / "processed"
        output_dir.mkdir(exist_ok=True)
        
        processed_ply = output_dir / "processed_cloud.ply"
        processor.save_processed_cloud(processed_ply)
        
        # Export to multiple formats
        processor.export_to_other_formats(output_dir)
        
        # Analyze point cloud
        analyzer = PointCloudAnalyzer(processor.point_cloud)
        density_info = analyzer.analyze_density()
        if density_info:
            print(f"Density analysis: {density_info}")
        
        # Create height map
        height_map = analyzer.create_height_map(resolution=0.1)
        if height_map is not None:
            print(f"Height map created with shape: {height_map.shape}")
        
        # Visualize (comment out if running headless)
        # processor.visualize(show_mesh=True)
        
        print(f"\nProcessing complete! Results saved to: {output_dir}")
    else:
        print(f"Dense point cloud not found at: {dense_ply}")
        print("Please run the COLMAP pipeline first!")