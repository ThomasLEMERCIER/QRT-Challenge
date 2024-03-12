import torch

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.act = torch.nn.SiLU()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x.to_numpy()).float()
        self.y = torch.tensor(y).long()
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

def train(model, optimizer, criterion, scheduler, train_dl, val_dl, n_epochs):
    best_acc = 0
    best_model = None
    device = next(model.parameters()).device
    
    for _ in range(n_epochs):
        model.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        acc = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                acc += (y_pred.argmax(dim=1) == y).sum().item()
            acc /= len(val_dl.dataset)
            if acc > best_acc:
                best_acc = acc
                best_model = model.state_dict()
    return best_model
