import numpy as np
import random 
import math
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from path import Path
import os
import json

from utils import read_off

# A executer dans PoinNet0_env

########################## TRANSFORMS ##############################
class PointSampler(object):
    def __init__(self, output_size, object=False):
        assert isinstance(output_size, int)
        self.output_size = output_size
        self.object = object
    
    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * ( side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s)*pt2[i] + (1-t)*pt3[i]
        return (f(0), f(1), f(2))
        
    
    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))


        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                        verts[faces[i][1]],
                                        verts[faces[i][2]]))
            
        sampled_faces = (random.choices(faces, 
                                    weights=areas,
                                    cum_weights=None,
                                    k=self.output_size))
        
        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                verts[sampled_faces[i][1]],
                                                verts[sampled_faces[i][2]]))
        
        return sampled_points


# class PointSampler(object):
#     def __init__(self, output_size):
#         assert isinstance(output_size, int)
#         self.output_size = output_size
#         self.object = object
    
#     def triangle_area(self, pt1, pt2, pt3):
#         side_a = np.linalg.norm(pt1 - pt2)
#         side_b = np.linalg.norm(pt2 - pt3)
#         side_c = np.linalg.norm(pt3 - pt1)
#         s = 0.5 * (side_a + side_b + side_c)
#         return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

#     def sample_point(self, pt1, pt2, pt3):
#         # barycentric coordinates on a triangle
#         # https://mathworld.wolfram.com/BarycentricCoordinates.html
#         s, t = sorted([random.random(), random.random()])
#         f = lambda i: s * pt1[i] + (t-s)*pt2[i] + (1-t)*pt3[i]
#         return (f(0), f(1), f(2))
        
    
#     def __call__(self, mesh):
#         verts, faces = mesh
#         verts = np.array(verts)
#         areas = np.zeros((len(faces)))

#         # Calculate areas of all triangles
#         for i in range(len(areas)):
#             areas[i] = self.triangle_area(verts[faces[i][0]],
#                                           verts[faces[i][1]],
#                                           verts[faces[i][2]])

#         # Sample faces according to their areas
#         sampled_faces = random.choices(faces, weights=areas, k=self.output_size)

#         sampled_points = np.zeros((self.output_size, 3))
#         sampled_faces_indices = []  # To store the face connections of the sampled points
        
#         # Map from original face index to sampled point index
#         sampled_points_map = {}
#         current_idx = 0  # To track the index of the new point cloud

#         for i, face in enumerate(sampled_faces):
#             # Get the original vertices of the face
#             pt1_idx, pt2_idx, pt3_idx = face
#             pt1, pt2, pt3 = verts[pt1_idx], verts[pt2_idx], verts[pt3_idx]

#             # Sample a point on the triangle
#             sampled_point = self.sample_point(pt1, pt2, pt3)
#             sampled_points[i] = sampled_point

#             # Check if each vertex of the face is already in the map, if not, add it
#             if pt1_idx not in sampled_points_map:
#                 sampled_points_map[pt1_idx] = current_idx
#                 current_idx += 1

#             if pt2_idx not in sampled_points_map:
#                 sampled_points_map[pt2_idx] = current_idx
#                 current_idx += 1

#             if pt3_idx not in sampled_points_map:
#                 sampled_points_map[pt3_idx] = current_idx
#                 current_idx += 1

#             # Store the face indices for the new sampled points
#             face_indices = [
#                 sampled_points_map[pt1_idx],
#                 sampled_points_map[pt2_idx],
#                 sampled_points_map[pt3_idx]
#             ]
#             sampled_faces_indices.append(face_indices)
            
        
#         with open('outputs/idx_faces.json', 'w') as f:
#             json.dump({'idx_faces':list(sampled_faces_indices)}, f, indent=4)

#         return sampled_points

        
    

class Normalize(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        
        norm_pointcloud = pointcloud - np.mean(pointcloud, axis=0) 
        norm_pointcloud /= np.max(np.linalg.norm(norm_pointcloud, axis=1))

        return  norm_pointcloud
    
class RandRotation_z(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), -math.sin(theta),    0],
                               [ math.sin(theta),  math.cos(theta),    0],
                               [0,                             0,      1]])
        
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return  rot_pointcloud
    
class RandomNoise(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        noise = np.random.normal(0, 0.02, (pointcloud.shape))
    
        noisy_pointcloud = pointcloud + noise
        return  noisy_pointcloud


# ######################
# class RandRotation_zv2(object):
#     def __init__(self, theta):
#         self.theta = theta    
    
#     def __call__(self, pointcloud):
#         assert len(pointcloud.shape)==2

#         #theta = random.random() * 2. * math.pi
#         rot_matrix = np.array([[ math.cos(self.theta), -math.sin(self.theta),    0],
#                                [ math.sin(self.theta),  math.cos(self.theta),    0],
#                                [0,                             0,      1]])
        
#         rot_pointcloud = rot_matrix.dot(pointcloud.T).T
#         return  rot_pointcloud
    
# class RandomNoise_v2(object):
#     def __init__(self, noise):
#         self.noise = noise    
        
#     def __call__(self, pointcloud):
#         assert len(pointcloud.shape)==2

#         #noise = np.random.normal(0, 0.02, (pointcloud.shape))
    
#         noisy_pointcloud = pointcloud + self.noise
#         return  noisy_pointcloud
 ########################""   
    
class ToTensor(object):
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        return torch.from_numpy(pointcloud)
    
    
def default_transforms():
    return transforms.Compose([
                                PointSampler(1024),
                                Normalize(),
                                ToTensor()
                              ])
    
    
########################## DATASET CLASS ##############################
class PointCloudData(Dataset):
    def __init__(self, root_dir, valid=False, folder="train", transform=default_transforms()):
        self.root_dir = root_dir
        folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.files = []
        for category in self.classes.keys():
            new_dir = root_dir/Path(category)/folder
            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {}
                    sample['pcd_path'] = new_dir/file
                    sample['category'] = category
                    #sample['filename'] = file
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file):
        verts, faces = read_off(file)
        if self.transforms:
            pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        #filename = pcd_path.name  # Get the filename
        
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f)
            
        return {'pointcloud': pointcloud, 
                'category': self.classes[category],
                'filename': pcd_path.name  # Return the filename
        }
        
        
########################## MODEL ##############################import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Tnet(nn.Module):
   def __init__(self, k=3):
      super().__init__()
      self.k=k
      self.conv1 = nn.Conv1d(k,64,1)
      self.conv2 = nn.Conv1d(64,128,1)
      self.conv3 = nn.Conv1d(128,1024,1)
      self.fc1 = nn.Linear(1024,512)
      self.fc2 = nn.Linear(512,256)
      self.fc3 = nn.Linear(256,k*k)

      self.bn1 = nn.BatchNorm1d(64)
      self.bn2 = nn.BatchNorm1d(128)
      self.bn3 = nn.BatchNorm1d(1024)
      self.bn4 = nn.BatchNorm1d(512)
      self.bn5 = nn.BatchNorm1d(256)
       

   def forward(self, input):
      # input.shape == (bs,n,3)
      bs = input.size(0)
      xb = F.relu(self.bn1(self.conv1(input)))
      xb = F.relu(self.bn2(self.conv2(xb)))
      xb = F.relu(self.bn3(self.conv3(xb)))
      pool = nn.MaxPool1d(xb.size(-1))(xb)
      flat = nn.Flatten(1)(pool)
      xb = F.relu(self.bn4(self.fc1(flat)))
      xb = F.relu(self.bn5(self.fc2(xb)))
      
      #initialize as identity
      init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)
      if xb.is_cuda:
        init=init.cuda()
      matrix = self.fc3(xb).view(-1,self.k,self.k) + init
      return matrix


class Transform(nn.Module):
   def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(3,64,1)

        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
       

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
       
   def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)

        xb = F.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)

        xb = F.relu(self.bn2(self.conv2(xb)))
        per_point_features = self.bn3(self.conv3(xb))  # Extract per-point features before pooling
        xb = nn.MaxPool1d(per_point_features.size(-1))(per_point_features)
        output = nn.Flatten(1)(xb)
        return output, per_point_features, matrix3x3, matrix64x64

class PointNet(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        self.transform = Transform()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)
        

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        xb, per_point_features, matrix3x3, matrix64x64 = self.transform(input)
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        output = self.fc3(xb)
        return self.logsoftmax(output), per_point_features, matrix3x3, matrix64x64
    
    
def pointnetloss(outputs, labels, m3x3, m64x64, alpha = 0.0001):
    criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
        id64x64=id64x64.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)

########################## TRAINING ##############################
def train(model, train_loader, device, opti, val_loader=None,  epochs=15, save=True):
    for epoch in range(epochs): 
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            opti.zero_grad()
            outputs, m3x3, m64x64 = model(inputs.transpose(1,2))

            loss = pointnetloss(outputs, labels, m3x3, m64x64)
            loss.backward()
            opti.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                    print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                        (epoch + 1, i + 1, len(train_loader), running_loss / 10))
                    running_loss = 0.0

        model.eval()
        correct = total = 0

        # validation
        if val_loader:
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                    outputs, __, __ = model(inputs.transpose(1,2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_acc = 100. * correct / total
            print('Valid accuracy: %d %%' % val_acc)

        # save the model
        if save:
            torch.save(model.state_dict(), "save_"+str(epoch)+".pth")