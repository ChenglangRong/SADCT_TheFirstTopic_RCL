import random
import numpy as np
import collections
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from clustertool.SACT_ConcurrentProcessing_ENV1 import Environment
import warnings
from pylab import mpl


# -------------------------------------------------
#   Q网络
# -------------------------------------------------
class Qnet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.q = torch.nn.Linear(hidden_dim,action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        # x = torch.relu(self.fc3(x))
        q = self.q(x)
        return q

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))