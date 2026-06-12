from torch import nn
from params import model_param as mp


class Discriminator(nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()

        self.restored = False
        self.layer = nn.Sequential(
            nn.Linear(mp.d_input_dims, mp.d_hidden_dims),
            nn.ReLU(),
            nn.Linear(mp.d_hidden_dims, mp.d_hidden_dims),
            nn.ReLU(),
            nn.Linear(mp.d_hidden_dims, mp.d_output_dims)
        )

    def forward(self, input):

        out = self.layer(input)
        return out
