import numpy as np
import torch

def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces

def write_obj_with_colors_from_pt_cloud(pt_cloud, idx_critical_pts, file_name, path_outputs_folder = "", kind_of_outputs = {'critical and non-critical points' : True, 'only critical points' : True, 'objet' : True}):
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
        print(f"OBJ file saved to {path_outputs_folder+'critical_pts_and_'+file_name+'.obj'}")
        
    # QUE les Critical points : len(idx_critical_pts) pts
    if kind_of_outputs['only critical points']:
        with open(path_outputs_folder+'critical_pts_OF_'+file_name+'.obj', 'w') as f:
            for idx, (x, y, z) in enumerate(pt_cloud):
                if idx in idx_critical_pts:
                    # Red color for critical points (1.0 0.0 0.0)
                    f.write(f"v {x.item()} {y.item()} {z.item()} 1.0 0.0 0.0\n")
        print(f"OBJ file saved to {path_outputs_folder+'critical_pts_of_'+file_name+'.obj'}")
        
    # TOUT les pts SANS DISTINCTION : 1024 pts
    if kind_of_outputs['only critical points']:
        with open(path_outputs_folder+'all_pts_OF_'+file_name+'.obj', 'w') as f:
            for idx, (x, y, z) in enumerate(pt_cloud):
                f.write(f"v {x.item()} {y.item()} {z.item()} 0.0 0.0 0.0\n")
        print(f"OBJ file saved to {path_outputs_folder+'all_pts_OF_'+file_name+'.obj'}")
        
  
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



