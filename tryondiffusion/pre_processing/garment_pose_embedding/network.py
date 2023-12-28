import torch.nn as nn


class AutoEncoder(nn.Module):

    def __init__(self, inp):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(inp, 16),
            nn.ReLU(True),
            nn.Linear(16, 8),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(True),
            nn.Linear(16, inp),
            nn.Sigmoid()
        )

    def forward(self, x):
        embeddings = self.encoder(x)
        prediction = self.decoder(embeddings)
        return prediction, embeddings


if __name__=="__main__":
    net = AutoEncoder(20)
    print(net)
