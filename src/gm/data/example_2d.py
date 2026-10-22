import torch
from sklearn.datasets import make_moons

def sample_data(batch_size,):
    X, y = make_moons(batch_size, noise=0.1)
    return torch.from_numpy(X).float()

def sample_noise(batch_size, n_modes=8, radius=2.0, std=0.1):
    angles = torch.randint(0, n_modes, (batch_size,)).float() * (2 * 3.14159265 / n_modes)
    centers = radius * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) + 0.5
    return centers + std * torch.randn(batch_size, 2)