"""
deep_models.py
--------------
Deep architectures from Section 3.4 of the paper, in PyTorch (CPU):

  DBN  -- stack of Restricted Boltzmann Machines, greedily pre-trained with
          Contrastive Divergence (CD-1), weights transferred into a feed-forward
          net and fine-tuned with back-propagation (Section 3.4.1).
  CNN  -- 1-D feature vector reshaped into a (recent-transactions x feature)
          matrix and passed through conv/pool/fc layers (Section 3.4.2).
  RNN  -- Elman recurrent network over the chronological transaction sequence
          (Section 3.4.3, Fig. 5).
"""
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
DEVICE = "cpu"


# ----------------------------------------------------------------- RBM ------
class RBM:
    """Bernoulli-Bernoulli RBM trained with CD-1 (inputs scaled to [0,1])."""
    def __init__(self, n_vis, n_hid, lr=0.05, epochs=8, batch=256):
        self.W = 0.01 * torch.randn(n_vis, n_hid)
        self.vb = torch.zeros(n_vis)
        self.hb = torch.zeros(n_hid)
        self.lr, self.epochs, self.batch = lr, epochs, batch

    def _ph(self, v):
        return torch.sigmoid(v @ self.W + self.hb)

    def _pv(self, h):
        return torch.sigmoid(h @ self.W.t() + self.vb)

    def fit(self, X):
        X = torch.as_tensor(X, dtype=torch.float32)
        n = X.shape[0]
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch):
                v0 = X[perm[i:i + self.batch]]
                ph0 = self._ph(v0)
                h0 = (ph0 > torch.rand_like(ph0)).float()
                pv1 = self._pv(h0)
                ph1 = self._ph(pv1)
                self.W += self.lr * (v0.t() @ ph0 - pv1.t() @ ph1) / v0.shape[0]
                self.vb += self.lr * (v0 - pv1).mean(0)
                self.hb += self.lr * (ph0 - ph1).mean(0)
        return self

    def transform(self, X):
        return self._ph(torch.as_tensor(X, dtype=torch.float32)).numpy()


class DBN(nn.Module):
    def __init__(self, n_in, hidden=(128, 64)):
        super().__init__()
        layers, prev = [], n_in
        self.blocks = nn.ModuleList()
        for h in hidden:
            self.blocks.append(nn.Linear(prev, h)); prev = h
        self.out = nn.Linear(prev, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        for b in self.blocks:
            x = self.act(b(x))
        return self.out(x).squeeze(-1)

    def pretrain(self, X01, hidden=(128, 64)):
        """Greedy layer-wise RBM pre-training; copy weights into the net."""
        data = X01
        for blk, h in zip(self.blocks, hidden):
            rbm = RBM(data.shape[1], h).fit(data)
            with torch.no_grad():
                blk.weight.copy_(rbm.W.t())
                blk.bias.copy_(rbm.hb)
            data = rbm.transform(data)
        return self


class CNNNet(nn.Module):
    """Input: (batch, 1, T, F) feature matrix of recent transactions."""
    def __init__(self, T, F):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(2, 3), padding=(0, 1)), nn.ReLU(),
            nn.AdaptiveMaxPool2d((2, 8)),
            nn.Conv2d(16, 32, kernel_size=(2, 3), padding=(0, 1)), nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 4)),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(nn.Linear(32 * 4, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.fc(self.net(x)).squeeze(-1)


class RNNNet(nn.Module):
    """Elman RNN over the last-T transaction feature vectors."""
    def __init__(self, F, hidden=64):
        super().__init__()
        self.rnn = nn.RNN(F, hidden, batch_first=True, nonlinearity="tanh")
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):                       # x: (batch, T, F)
        o, _ = self.rnn(x)
        return self.fc(o[:, -1, :]).squeeze(-1)


# ------------------------------------------------------------- training -----
def train_torch(model, Xtr, ytr, epochs=12, batch=512, lr=1e-3, pos_weight=None):
    model.to(DEVICE)
    Xtr = torch.as_tensor(Xtr, dtype=torch.float32)
    ytr = torch.as_tensor(ytr, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pw = torch.tensor([pos_weight]) if pos_weight else None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    n = Xtr.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            out = model(Xtr[idx])
            loss = loss_fn(out, ytr[idx])
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def predict_proba_torch(model, X, batch=2048):
    model.eval()
    X = torch.as_tensor(X, dtype=torch.float32)
    outs = []
    for i in range(0, X.shape[0], batch):
        outs.append(torch.sigmoid(model(X[i:i + batch])).cpu().numpy())
    return np.concatenate(outs)
