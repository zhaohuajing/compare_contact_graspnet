import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
height, width = 480, 640  # Depth map resolution
depth_range = [0.5, 2.0]  # Depth values in meters (min, max)
obj_center = [0.0, 0.0, 1.0]  # Object center in camera frame (x, y, z in meters)
obj_size = 0.1  # Cube size (edge length in meters)
num_obj_points = 1000  # Number of points in object point cloud
num_scene_points = 5000  # Number of points in scene point cloud

# Generate object point cloud (cube-like)
def generate_object_pc(center, size, num_points):
    # Generate points on a cube surface
    points = []
    for _ in range(num_points):
        face = np.random.randint(0, 6)  # Choose a cube face
        point = np.random.uniform(-size/2, size/2, 3)
        if face == 0: point[0] = size/2   # +X face
        elif face == 1: point[0] = -size/2  # -X face
        elif face == 2: point[1] = size/2   # +Y face
        elif face == 3: point[1] = -size/2  # -Y face
        elif face == 4: point[2] = size/2   # +Z face
        elif face == 5: point[2] = -size/2  # -Z face
        points.append(point + center)
    return np.array(points, dtype=np.float32)

# Generate scene point cloud (object + background plane)
def generate_scene_pc(obj_points, num_points):
    # Background plane at z=1.5 meters
    plane_z = 1.5
    plane_points = np.random.uniform([-0.5, -0.5, plane_z], [0.5, 0.5, plane_z], (num_points - len(obj_points), 3))
    # Combine object points and background points
    return np.concatenate([obj_points, plane_points], axis=0).astype(np.float32)

# Generate depth map
def generate_depth_map(height, width, obj_center, obj_size):
    depth = np.ones((height, width), dtype=np.float32) * 1.5  # Background depth
    # Project object cube into depth map using intrinsics
    fx, fy = 554.254691191187, 554.254691191187
    px, py = 320.5, 240.5
    # Define object boundaries in image plane
    obj_z = obj_center[2]
    x_min = int((obj_center[0] - obj_size/2) * fx / obj_z + px)
    x_max = int((obj_center[0] + obj_size/2) * fx / obj_z + px)
    y_min = int((obj_center[1] - obj_size/2) * fy / obj_z + py)
    y_max = int((obj_center[1] + obj_size/2) * fy / obj_z + py)
    x_min, x_max = max(0, x_min), min(width, x_max)
    y_min, y_max = max(0, y_min), min(height, y_max)
    # Set object depth
    depth[y_min:y_max, x_min:x_max] = obj_z
    return depth

# Generate data
obj_pc = generate_object_pc(obj_center, obj_size, num_obj_points)
scene_pc = generate_scene_pc(obj_pc, num_scene_points)
depth_map = generate_depth_map(height, width, obj_center, obj_size)

# Save to .npy files
np.save("obj_pc.npy", obj_pc)
np.save("scene_pc.npy", scene_pc)
np.save("depth.npy", depth_map)

print("Generated and saved:")
print(f" - obj_pc.npy: {obj_pc.shape} points")
print(f" - scene_pc.npy: {scene_pc.shape} points")
print(f" - depth.npy: {depth_map.shape} depth map")