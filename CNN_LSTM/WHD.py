from torch import nn
import numpy as np
import torch

class WHD(nn.Module):
    def __init__(self, dipole_pos, positions_per_trial, device):
        self.dipole_pos = dipole_pos
        self.positions_per_trial = positions_per_trial
        self.max_d, _ = self.max_euclidean_distance(self.dipole_pos)
        self.eps = 1e-6
        self.alpha=4
        self.device = device
    
    def WHD_loss(self, prob_map, idx):
        gt_points = torch.tensor(self.positions_per_trial.iloc[idx])
        n_est_pts = prob_map.sum()
        d_matrix = self.cdist(self.dipole_pos, gt_points)
        p_replicated = prob_map.view(-1,1).repeat(1, gt_points.shape[0])
        term_1 = (1 / (n_est_pts + self.eps)) * torch.sum(prob_map*torch.min(d_matrix, 1)[0])
        d_div_p = torch.min((d_matrix + self.eps) / (p_replicated**self.alpha + self.eps / self.max_d), 0)[0]
        d_div_p = torch.clamp(d_div_p, 0, self.max_d)
        term_2 = torch.mean(d_div_p, 0)
        return term_1 + term_2
    
    def cdist(self, x, y):
        '''
        Input: x is a Nxd Tensor
            y is a Mxd Tensor
        Output: dist is a NxM matrix where dist[i,j] is the norm
            between x[i,:] and y[j,:]
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||
        '''
        differences = x.unsqueeze(1) - y.unsqueeze(0)
        distances = torch.sum(differences**2, -1).sqrt()
        return distances.to(self.device, dtype=torch.float)
    
    def max_euclidean_distance(self, points):
        """
        Find the maximum Euclidean distance between any two points.
        
        Args:
            points: numpy array of shape (n, 3) containing 3D coordinates
        
        Returns:
            max_distance: float, the maximum distance
            indices: tuple (i, j), indices of the two points with max distance
        """
        # Compute all pairwise squared distances using broadcasting
        # diff shape: (n, n, 3)
        diff = points[:, torch.newaxis, :] - points[torch.newaxis, :, :]
        
        # Compute squared distances, shape: (n, n)
        sq_distances = torch.sum(diff**2, axis=2)
        
        # Find the maximum
        max_sq_dist = torch.max(sq_distances)
        max_distance = torch.sqrt(max_sq_dist)
        
        # Get indices of maximum distance
        i, j = torch.unravel_index(torch.argmax(sq_distances), sq_distances.shape)
        
        return max_distance, (i, j)