import torch
import torch.nn as nn
from tqdm import tqdm

class FlowMatchingModel(nn.Module):
    def __init__(self, dim = 2, hidden_dim = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, t, x_t):
        tx = torch.cat([t, x_t], dim=-1)
        return self.net(tx)

def interpolate_linear(x_0, x_1, t):
    x_t = ((1 - t) * x_0) + (t * x_1)
    return x_t

def get_target_velocity(x_0, x_1):
    return x_1 - x_0

def loss(fmm, x_0, x_1, t):
    x_t = interpolate_linear(x_0, x_1, t)
    v_target = get_target_velocity(x_0, x_1)
    v_pred = fmm(t, x_t)

    loss = ((v_pred - v_target)**2).mean()

    return loss


def train_fmm(model, optimizer, sample_noise, sample_data, steps, batch_size, device):
    losses = []
    with tqdm(range(steps)) as pb:
        for i in pb:
            x_0 = sample_noise(batch_size).to(device)
            x_1 = sample_data(batch_size).to(device)
            # t = torch.rand(x_1.shape[0], device=device).unsqueeze(-1)
            t = torch.rand(x_1.shape[0], device=device).view(-1, 1, 1, 1)
            
            optimizer.zero_grad()
            l = loss(model, x_0, x_1, t)
            l.backward()
            optimizer.step()

            pb.set_postfix({'Loss': f'{l.item():.2f}'})
            losses.append(l.item())

    return losses
    

@torch.no_grad()
def sample_fmm(fmm, x_0, n_steps = 100, device='cpu'):
    x = x_0
    dt = 1.0 / n_steps

    for i in range(n_steps):
        t_val = i * dt
        t = torch.full((x.shape[0], 1), t_val, device=device)
        v = fmm(t, x)
        x = x + v * dt

    return x