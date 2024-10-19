import numpy as np
import torch
import matplotlib.pyplot as plt


def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces

def write_obj_with_colors_from_pt_cloud(pt_cloud, idx_critical_pts, file_name, path_outputs_folder, kind_of_outputs = {'critical and non-critical points' : True, 'only critical points' : True, 'objet' : True}):
    """
    Writes a point cloud to an .obj file with specific points colored in red.
    Other points are colored in black.
    
    Args:
        pt_cloud (torch.Tensor): Tensor of shape (1, 1024, 3) containing the point cloud.
        idx_critical_pts (set): Indices of points to color red.
        output_file (str): Path of the output .obj file.
    """
    pt_cloud = pt_cloud.squeeze(0)  # Remove the first dimension to get shape (1024, 3)

    # Critical points ET les autres : 1024 pts
    if kind_of_outputs['critical and non-critical points']:
        with open(path_outputs_folder+'critical_pts_AND_'+file_name+'.obj', 'w') as f:
            for idx, (x, y, z) in enumerate(pt_cloud):
                if idx in idx_critical_pts:
                    # Red color for critical points (1.0 0.0 0.0)
                    f.write(f"v {x.item()} {y.item()} {z.item()} 1.0 0.0 0.0\n")
                else:
                    # Black color for non-critical points (0.0 0.0 0.0)
                    f.write(f"v {x.item()} {y.item()} {z.item()} 0.0 0.0 0.0\n")
        #print(f"OBJ file saved to {path_outputs_folder+'critical_pts_and_'+file_name+'.obj'}")
        
    # QUE les Critical points : len(idx_critical_pts) pts
    if kind_of_outputs['only critical points']:
        with open(path_outputs_folder+'critical_pts_OF_'+file_name+'.obj', 'w') as f:
            for idx, (x, y, z) in enumerate(pt_cloud):
                if idx in idx_critical_pts:
                    # Red color for critical points (1.0 0.0 0.0)
                    f.write(f"v {x.item()} {y.item()} {z.item()} 1.0 0.0 0.0\n")
        #print(f"OBJ file saved to {path_outputs_folder+'critical_pts_of_'+file_name+'.obj'}")
        
    # TOUT les pts SANS DISTINCTION : 1024 pts
    if kind_of_outputs['only critical points']:
        with open(path_outputs_folder+'all_pts_OF_'+file_name+'.obj', 'w') as f:
            for idx, (x, y, z) in enumerate(pt_cloud):
                f.write(f"v {x.item()} {y.item()} {z.item()} 0.0 0.0 0.0\n")
        #print(f"OBJ file saved to {path_outputs_folder+'all_pts_OF_'+file_name+'.obj'}")
        
  
def assign_gradient_color(point_cloud, indices_list):
    """
    Assign gradient colors to specific points in the point cloud based on their occurrences in indices_list.
    All other points will be colored black.

    Args:
        point_cloud (numpy.ndarray): Point cloud (Nx3) array where N is the number of points.
        indices_list (list): List of indices for which the occurrences need to be considered.

    Returns:
        numpy.ndarray: Array of colors corresponding to the points.
    """
    # Step 1: Count occurrences of each index in the list
    unique_indices, counts = np.unique(indices_list, return_counts=True)
    
    # Step 2: Normalize the counts to a range [0, 1]
    normalized_counts = (counts - counts.min()) / (counts.max() - counts.min())

    # Step 3: Create a colormap (e.g., from blue to red)
    cmap = plt.get_cmap('Reds')

    # Step 4: Initialize all points with black color (RGB = [0, 0, 0])
    point_colors = np.zeros((point_cloud.shape[0], 3))  # Nx3 array for RGB colors, initialized to black

    # Step 5: Assign colors to points in the indices_list based on their normalized counts
    for idx, count in zip(unique_indices, normalized_counts):
        point_colors[idx] = cmap(count)[:3]  # Assign RGB values from the colormap
        
    #print(normalized_counts, counts)
    return point_colors, counts
  

def write_obj_with_gradient_colors_from_pt_cloud(pt_cloud, pt_colors, file_name, path_outputs_folder = ""):
    """
    Write a point cloud with colors to an OBJ file.

    Args:
        point_cloud (numpy.ndarray): Array of 3D coordinates (Nx3).
        point_colors (numpy.ndarray): Array of RGB colors for each point (Nx3), with values between 0 and 1.
        file_name (str): Output OBJ file name (with .obj extension).
    """
    if pt_cloud.shape[0] != pt_colors.shape[0]:
        raise ValueError("Point cloud and point colors must have the same number of points.")
    
    # Open the file in write mode
    with open(path_outputs_folder+'gradient_color_critical_pts_AND_'+file_name+'.obj', 'w') as file:
        # Write vertices with colors
        for i in range(pt_cloud.shape[0]):
            x, y, z = pt_cloud[i]
            r, g, b = pt_colors[i]
            # Write the vertex with color in OBJ format (v x y z r g b)
            file.write(f"v {x} {y} {z} {r} {g} {b}\n")

    #print(f"OBJ file '{file_name}' with gradient colors written successfully.")

  
### ATTENTION 
# indice face commence à 1 pour OBJ 
# indice face commence à 0 pour OFF     
def write_off_file(filename, vertices, faces):
    """
    Write vertices and faces to an .obj file.
    
    Args:
        filename (str): The path of the output .obj file.
        vertices (arrays): A tensor of shape (N, 3) containing the 3D coordinates of N vertices.
        faces (list of lists or array): A list or array of shape (M, 3) where each element is a list/array
                                        containing the indices of the vertices forming each face.
    """
    print(f"----- ATTENTION ------\n\
        # indice face commence à 1 pour OBJ \n\
        # indice face commence à 0 pour OFF     ")  
    with open(filename, 'w') as file:
        # Write the vertices
        for vertex in vertices:
            file.write(f"v {vertex[0].item()} {vertex[1].item()} {vertex[2].item()}\n")
        
        # Write the faces (OBJ uses 1-based indexing, so we add 1 to the indices)
        for face in faces:
            file.write(f"f {face[0]} {face[1]} {face[2]}\n")
                    
### ATTENTION 
# indice face commence à 1 pour OBJ 
# indice face commence à 0 pour OFF    
def write_mesh(vertices, faces, filename):
    """
    Write a 3D mesh to either an OBJ or OFF file based on the minimum face index.

    Args:
        vertices (np.ndarray or torch.Tensor): Array or tensor of vertex coordinates (shape: [N, 3]).
        faces (np.ndarray): Array of face indices (shape: [M, 3] or [M, 4] for quads).
        filename (str): Name of the output file.
    """
    
    # Check if vertices is a tensor and convert to numpy array
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.cpu().numpy()

    # Determine if it's an OBJ or OFF file by checking the min value of faces
    if np.min(faces) == 1:
        # Write OBJ file (1-based indexing)
        with open(filename + '.obj', 'w') as f:
            # Write vertices
            for vertex in vertices:
                f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

            # Write faces (OBJ format is 1-based)
            for face in faces:
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")
        print(f"OBJ file written to {filename}.obj")
    
    elif np.min(faces) == 0:
        # Write OFF file (0-based indexing)
        with open(filename + '.off', 'w') as f:
            f.write("OFF\n")
            f.write(f"{len(vertices)} {len(faces)} 0\n")  # num of vertices, faces, edges (optional)

            # Write vertices
            for vertex in vertices:
                f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

            # Write faces (OFF format is 0-based)
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        print(f"OFF file written to {filename}.off")
    else:
        print("Error: Invalid face indices. Faces should start from 0 or 1.")



