import os
import argparse
from path import Path
import json
import pickle

from pointnet_functions import *
from utils import *

#######################################
# A executer dans PoinNet0_env
#######################################
# Ligne de commande : python3 get_critical_points.py /.../ModelNet10/chair/test/chair_0896.off outputs

# python get_critical_points.py /.../ModelNet10/chair/test/chair_0948.off outputs --kind_of_outputs '{"critical and non-critical points": true, "only critical points": false, "objet": false}'


# Function to parse arguments
def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process different types of files.")
    
    parser.add_argument('input_path', type=str, help='Path to the specific input file')
    parser.add_argument('output_path', type=str, help='Path to the output directory')
    parser.add_argument('--kind_of_outputs', type=str, default='{"critical and non-critical points": true, "only critical points": true, "objet": true}', help='JSON string representing the type of outputs')
    
    return parser.parse_args()

# Function to setup the dataset
def setup_dataset(path_to_file):
    # Define the transformations
    train_transforms = transforms.Compose([
                    PointSampler(1024),
                    Normalize(),
                    RandRotation_z(),
                    RandomNoise(),
                    ToTensor()
                    ])
    
    # Read the .off file + apply the transformations
    with open(path_to_file, 'r') as object:
        verts, faces = read_off(object)
        ptcloud = train_transforms((verts, faces))
        
    return ptcloud

def save_point_cloud_info(input_path, pcd, idx_critical_points, occur, file_name, output_dir):
    """
    Save the point cloud information (metadata and pcd data).
    
    Args:
        pcd (torch.Tensor): Tensor containing the point cloud data.
        idx_critical_points (list): Indices of critical points in the point cloud.
        file_name (str): The original name of the point cloud file.
        output_dir (str): Directory where the data will be saved.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create metadata
    point_cloud_info = {
        'input_path': input_path,
        'folder': Path(input_path).parent.name,
        'file_name': file_name,
        'category' : file_name.split('_')[0],
        'numero' : file_name.split('_')[1],
        'nb_critical_points': len(idx_critical_points),
        'shape': pcd.squeeze(0).numpy().shape,
        'critical_points_indices': list(idx_critical_points),        
        'num_points': pcd.squeeze(0).tolist() ,
        'occurences': occur.tolist()
    }
    
    # Save the metadata as a JSON file
    metadata_file = os.path.join(output_dir, f"{file_name}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(point_cloud_info, f, indent=4)
    
    print(f"Metadata saved to {metadata_file}")


# Main function
def main():
    # Parse command-line arguments
    args = parse_args()
    
    input_path = args.input_path
    name_file = Path(input_path).name.split('.')[0]
    output_path = args.output_path
    # Convert the kind_of_outputs argument from a JSON string to a dictionary
    dict_kind_of_outputs = json.loads(args.kind_of_outputs)
    
    pointnet = PointNet()
    pointnet.load_state_dict(torch.load('save.pth', map_location=torch.device('cpu')))  # or use 'cuda' if using GPU
    pointnet.eval();
    
    # Point cloud of the objet 
    points_cloud = setup_dataset(input_path)
    pt_cloud_reshape = points_cloud.unsqueeze(0).float(); #print(pt_cloud_reshape.shape, pt_cloud_reshape.dtype)

    # Assuming 'point_cloud' has shape [B, 3, N], where B is batch size, 3 are coordinates, N is number of points
    _, per_point_features, _, _ = pointnet(pt_cloud_reshape.transpose(1,2))

    # Apply max pooling to get the global feature and indices of critical points
    _, indices = torch.max(per_point_features, dim=2)

    # Extract critical points from the original point cloud
    idx_critical_points = set([item for sublist in indices.tolist() for item in sublist]); len(idx_critical_points)
    
    print(f"\nName of file: {name_file}")
    print(f"Output path: {output_path}")
    print(f"Kind of outputs: {dict_kind_of_outputs}\n")
    
    # Write the .obj file
    write_obj_with_colors_from_pt_cloud(pt_cloud_reshape, idx_critical_points, name_file, "outputs/", dict_kind_of_outputs)
    
    # Gradient color : Write the .obj file
    point_colors, occurences = assign_gradient_color(pt_cloud_reshape.squeeze(0).numpy(), indices.tolist()[0])
    write_obj_with_gradient_colors_from_pt_cloud(pt_cloud_reshape.squeeze(0).numpy(), point_colors, name_file, "outputs/")
    
    # Save the dictionary to a JSON file
    save_point_cloud_info(input_path, pt_cloud_reshape, idx_critical_points, occurences, name_file, output_path)
    
    
    


    
if __name__ == '__main__':
    main()
