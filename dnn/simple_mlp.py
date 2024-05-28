import torch
import torch.nn as nn


class MultiLayerPerception(nn.Module):

    def __init__(self):
        super(MultiLayerPerception, self).__init__()
        
        self.input_layer = nn.Linear(3, 512)
        hidden_layer = [nn.Linear(512, 512) for _ in range(2)]
        self.hidden_layer = nn.ModuleList(hidden_layer)
        self.output_layer = nn.Linear(512, 3)
        with torch.no_grad():
            self.output_layer.weight.fill_(0)
            self.output_layer.bias.fill_(0)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layer:
            x = torch.tanh(layer(x))
        x = self.output_layer(x)

        return x