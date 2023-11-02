import torch
from torch import nn

class Net(nn.Module):
    def __init__(self, n_actions, dueling=False):
        super(Net, self).__init__()
        self.n_actions = n_actions
        self.dueling = dueling

        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, (1, 1), stride=(1, 1), padding=(0, 0)), nn.ReLU(), nn.BatchNorm2d(16))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, (1, 2), stride=(1, 1), padding=(0, 0)), nn.ReLU(), nn.BatchNorm2d(32))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, (1, 3), stride=(1, 1), padding=(0, 0)), nn.ReLU(), nn.BatchNorm2d(64))
        # output: 4*10*64 for N_J = 20, N_F = 4
        # output: 8*14*64 for N_J = 40, N_F = 8

        if not dueling:
            self.value = nn.Sequential(
                nn.Linear(8*14*64, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, n_actions),
            )
        else:
            self.value = nn.Sequential(
                nn.Linear(8*14*64, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            self.advantage = nn.Sequential(
                nn.Linear(8*14*64, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, n_actions),
            )
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.flatten().reshape((batch_size, -1))

        if not self.dueling:
            q_values = self.value(x)
            return q_values
        else:
            # advantage
            ya = self.advantage(x)
            mean = torch.reshape(torch.sum(ya, dim=1) / self.n_actions, (batch_size, 1))
            ya, mean = torch.broadcast_tensors(ya, mean)
            ya -= mean

            # state value
            ys = self.value(x)
            ya, ys = torch.broadcast_tensors(ya, ys)
            action_values = ya + ys
            return action_values