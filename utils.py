def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces

def write_obj_with_colors(pt_cloud, idx_critical_pts, file_name, path_outputs_folder = "", kind_of_outputs = {'critical and non-critical points' : True, 'only critical points' : True, 'objet' : True}):
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


