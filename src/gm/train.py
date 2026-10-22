import torch
import numpy as np
from tqdm import tqdm

from src.gm.data.example_2d import sample_data, sample_noise
from src.gm.models.flow_matching_model import FlowMatchingModel, train_fmm
from src.gm.models.energy_based_model import EnergyBasedModel, train_ebm
from src.gm.models.energy_matching_model import EnergyMatchingModel, train_emm


D = 2
N = 2_000
SEED = 42
DEVICE = 'mps'

steps = 500
batch_size = 256
lr = 1e-3
eps_max = 0.05

emm_phase1_steps = 500
emm_phase2_steps = 500
cd_weight = 1.0

np.random.seed(SEED)
torch.manual_seed(SEED)

fmm = FlowMatchingModel().to(DEVICE)
opt = torch.optim.Adam(fmm.parameters(), lr)
train_fmm(fmm, opt, sample_noise, sample_data, steps, batch_size, DEVICE)

ebm = EnergyBasedModel().to(DEVICE)
opt = torch.optim.Adam(ebm.parameters(), lr)
train_ebm(ebm, opt, sample_noise, sample_data, steps, batch_size, eps_max, DEVICE)

emm = EnergyMatchingModel().to(DEVICE)
opt = torch.optim.Adam(emm.parameters(), lr)
train_emm(emm, opt, sample_noise, sample_data, emm_phase1_steps, emm_phase2_steps, batch_size, eps_max, cd_weight, DEVICE)