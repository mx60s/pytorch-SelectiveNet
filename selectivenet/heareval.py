import torch.nn as nn
import torch.nn.functional as F

class HearEvalNN(nn.Module):
    """
    The fully connected predictor model used on top of hubert embeddings, as calculated by hear benchmarks
    """
    def __init__(self, input_dim=1024):
        super(HearEvalNN, self).__init__()
        self.layers = nn.ModuleList()

        hidden_dim = 1024
        hidden_layers = 2
        dropout = 0.1
        
        for i in range(hidden_layers):
            self.layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.Dropout(dropout))

        self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))

        return x
