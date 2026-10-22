import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

class EnergyMatchingModel(nn.Module):
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
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

def grad_V(V, x):
    # x = x.detach().requires_grad_(True)
    energy = V(x).sum()
    (g,) = torch.autograd.grad(energy, x, create_graph=True)
    return g

def ot_couple(x0, x1):
    cost = torch.cdist(x0.flatten(1), x1.flatten(1)) ** 2
    row_ind, col_ind = linear_sum_assignment(cost.cpu().detach().numpy())

    return x0[row_ind], x1[col_ind]

def phase1_loss(V, x_0, x_1, t):
    x_0, x_1 = ot_couple(x_0, x_1)

    x_t = (1 - t) * x_0 + t * x_1
    target = x_0 - x_1
    pred = grad_V(V, x_t)

    return ((pred - target) ** 2).sum(dim=1).mean()

def langevin_negative_samples(V, x_int, n_steps = 20, step_size = 0.01, eps_max = 0.05):
    x = x_int.clone().detach()
    for _ in range(n_steps):
        g = grad_V(V, x).detach()
        noise = torch.randn_like(x)

        x = x - step_size * g + (2 * step_size * eps_max) ** 0.5 * noise
        x = x.detach()

    return x

def ode_integrate(V, x, n_steps = 50, dt = None):
    dt = dt if dt is not None else 1.0 / n_steps
    x = x.clone().detach()
    for _ in range(n_steps):
        g = grad_V(V, x).detach()
        x = x - dt * g

    return x

def phase2_loss(V, x_0, x_1, t, eps_max = 0.05, cd_weight = 1.0):
    ot_loss = phase1_loss(V, x_0, x_1, t)

    half = x_1.shape[0] // 2
    x_0 = x_0[:half]
    x0_pushed = ode_integrate(V, x_0, n_steps=10)
    x_init = torch.cat([x_1[:half] + 0.1 * torch.randn_like(x_1[:half]), x0_pushed], dim=0)

    x_neg = langevin_negative_samples(V, x_init, eps_max=eps_max)

    cd_loss = V(x_1).mean() / eps_max - V(x_neg).mean() / eps_max
    return ot_loss + cd_weight * cd_loss


def train_emm(model, optimizer, sample_noise, sample_data, phase1_steps, phase2_steps, batch_size, eps_max = 0.05, cd_weight = 1.0, device = 'cpu'):
    phase1_losses = []
    phase2_losses = []

    with tqdm(range(phase1_steps)) as pb:
        for i in pb:
            x_0 = sample_noise(batch_size).to(device)
            x_1 = sample_data(batch_size).to(device)
            t = torch.rand(x_0.shape[0], 1, 1, 1, device=device)

            optimizer.zero_grad()
            loss = phase1_loss(model, x_0, x_1, t)
            loss.backward()
            optimizer.step() 

            pb.set_postfix({'Loss': f'{loss.item():.2f}'})
            phase1_losses.append(loss.item())

    with tqdm(range(phase2_steps)) as pb:
        for i in pb:
            x_0 = sample_noise(batch_size).to(device)
            x_1 = sample_data(batch_size).to(device)
            t = torch.rand(x_0.shape[0], 1, 1, 1, device=device)

            optimizer.zero_grad()
            loss = phase2_loss(model, x_0, x_1, t, eps_max, cd_weight)
            loss.backward()
            optimizer.step() 

            pb.set_postfix({'Loss': f'{loss.item():.2f}'})
            phase2_losses.append(loss.item())

    return phase1_losses, phase2_losses

@torch.no_grad()
def sample_emm(V, x_0, ode_steps = 50, langevin_steps = 10, eps_max = 0.05):
    x = x_0
    with torch.enable_grad():
        x = ode_integrate(V, x, n_steps=ode_steps)
        x = langevin_negative_samples(V, x, n_steps=langevin_steps, eps_max=eps_max)

    return x