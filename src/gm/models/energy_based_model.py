import torch
import torch.nn as nn
from tqdm import tqdm

class EnergyBasedModel(nn.Module):
    def __init__(self, dim = 2, hidden_dim = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)

def grad_V(V, x):
    x = x.detach().requires_grad_(True)
    energy = V(x).sum()
    (g, ) = torch.autograd.grad(energy, x, create_graph=True)
    return g

def langevin_negative_samples(V, x_init, n_steps = 50, steps_size = 0.01, eps_max=0.05):
    x = x_init.clone().detach()
    for _ in range(n_steps):
        g = grad_V(V, x).detach()
        noise = torch.randn_like(x)

        x = x - steps_size * g + (2 * steps_size * eps_max) ** 0.5 * noise
        x = x.detach()

    return x

def loss(V, x_0, x_1, eps_max=0.05, alpha=0.1):
    x_neg = langevin_negative_samples(V, x_0, eps_max=eps_max)

    v_data = V(x_1)
    v_neg = V(x_neg)

    cd_loss = V(x_1).mean() / eps_max - V(x_neg).mean() / eps_max
    reg = alpha *  (v_data ** 2 + v_neg ** 2).mean()

    return cd_loss + reg

def train_ebm(model, optimizer, sample_noise, sample_data, steps, batch_size, eps_max = 0.05, device = 'cpu'):
    losses = []
    with tqdm(range(steps)) as pb:
        for i in pb:
            x_0 = sample_noise(batch_size).to(device)
            x_1 = sample_data(batch_size).to(device)
            
            optimizer.zero_grad()
            l = loss(model, x_0, x_1, eps_max)
            l.backward()
            optimizer.step()

            pb.set_postfix({'Loss': f'{l.item():.2f}'})
            losses.append(l.item())

    return losses


@torch.no_grad()
def sample_ebm(V, x_0, eps_max=0.05):
    x = x_0
    with torch.enable_grad():
        x = langevin_negative_samples(V, x, eps_max=eps_max)

    return x