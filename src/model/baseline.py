import torch.nn as nn
import torch.nn.functional as F


class BaseCNN(nn.Module):
    def __init__(self, input_shape):
        super(BaseCNN, self).__init__()
        c, h, w = input_shape
        self.conv_1 = nn.Conv2d(in_channels=c, out_channels=4, kernel_size=5, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
        self.conv2_out_size = 10 * 4 * 4 if c == 1 else 10 * 5 * 5
        self.fc_1 = nn.Linear(in_features=self.conv2_out_size, out_features=100)
        self.fc_2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, self.conv2_out_size)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x
