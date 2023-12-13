import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt
import random
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToPILImage

sys.path.append('../..')
from tllib.alignment.regda import PoseResNet2d as RegDAPoseResNet, \
    PseudoLabelGenerator2d, RegressionDisparity
import tllib.vision.models as models
from tllib.vision.models.keypoint_detection.pose_resnet import Upsampling, PoseResNet
from tllib.vision.models.keypoint_detection.loss import JointsKLLoss
import tllib.vision.datasets.keypoint_detection as datasets
import tllib.vision.transforms.keypoint_detection as T
from tllib.vision.transforms import Denormalize
from tllib.utils.data import ForeverDataIterator
from tllib.utils.meter import AverageMeter, ProgressMeter, AverageMeterDict
from tllib.utils.metric.keypoint_detection import accuracy
from tllib.utils.logger import CompleteLogger
from webcolors import name_to_rgb
device = torch.device("cuda:0")

# source_root = '/pfs/rdi/cei/synthetic_data/public_dataset/RHD/'
# source = 'RenderedHandPose' 
source_root = '/pfs/rdi/cei/synthetic_data/public_dataset/surreal/surreal_processed'
source = 'SURREAL'


image_size = (256,256)
heatmap_size = (64, 64)
resize_scale = (0.6, 1.3)
batch_size = 32


normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_transform = T.Compose([
    T.RandomRotation(60),
    T.RandomResizedCrop(size=image_size, scale=resize_scale),
    T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
    T.GaussianBlur(),
    T.ToTensor(),
    normalize
])
val_transform = T.Compose([
    T.Resize(image_size[0]),
    T.ToTensor(),
    normalize
])

# define visualization function
tensor_to_image = Compose([
    Denormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ToPILImage()
])

joint_names = ['hips','leftUpLeg','rightUpLeg', 'spine','leftLeg', 'rightLeg','spine1', 'leftFoot','rightFoot','spine2','leftToeBase',
            'rightToeBase','neck','leftShoulder','rightShoulder', 'head','leftArm', 'rightArm', 'leftForeArm', 'rightForeArm', 'leftHand', 
            'rightHand','leftHandIndex1', 'rightHandIndex1']
joint_names = np.array(joint_names)
joint_index =  (7, 4, 1, 2, 5, 8, 0, 9, 12, 15, 20, 18, 13, 14, 19, 21)
joint_names = list(joint_names[[joint_index]])



# define visualization function
tensor_to_image = Compose([
    Denormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ToPILImage()
])

source_dataset = datasets.__dict__[source]
train_source_dataset = source_dataset(root=source_root, split='train', transforms=val_transform,
                                    image_size=image_size, heatmap_size=heatmap_size)
train_source_loader = DataLoader(train_source_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
print("Source train:", len(train_source_loader))



import torch
import torch.nn as nn
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

# Hyperparameters
input_dim = 42
latent_dim = 256
lr = 1e-4
batch_size = 32
epochs = 60

# Create the VAE model
vae = VAE(input_dim, latent_dim)

# Define loss function
def loss_function(reconstructed_x, x, mu, logvar):
    reconstruction_loss = nn.functional.mse_loss(reconstructed_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + KLD

# Create optimizer
optimizer = optim.Adam(vae.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    total_loss = 0
    for  i, (index,x, label, weight, meta) in tqdm(enumerate(train_source_loader), leave = True, position = 0): 
        data_point = np.array(meta['keypoint2d']).astype(int)
        data_point = np.array(data_point).reshape(-1,32)
#         print(data_point.shape)
        data_point = torch.tensor(data_point, dtype=torch.float32)
        
        optimizer.zero_grad()
        reconstructed_data, mean, logvar = vae(data_point)
        loss = loss_function(reconstructed_data, data_point, mean, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss / 451000}')

# Generate samples from the trained VAE
# with torch.no_grad():
#     z = torch.randn(16, latent_dim)
#     generated_samples = vae.decoder(z)


anomalous_vae = []
errors = []
for i, (index, x, label, weight, meta) in tqdm(enumerate(train_source_loader), position=0, leave=True):
    data_points = np.array(meta['keypoint2d']).astype(int).reshape(-1, 32)
    data_points = torch.tensor(data_points, dtype=torch.float32)
    for dj in range(len(data_points)):
        reconstructed_data, _, _ = vae(data_points[dj])
        reconstruction_error = torch.norm(data_points[dj] - reconstructed_data).item()
#         print(reconstruction_error)
        errors.append(reconstruction_error)
        anomalous_vae.append(index[dj])

x = np.array(errors)
y = np.array(anomalous_vae)

np.save('indexes_vae_poses_REAL_H36M_TRAIN.npy',y)
np.save('recons_errors_vae_REAL_H36M_TRAIN.npy',x)
torch.save(vae, './hpe_real_H36M_train_vae_model')
