import torch

class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(MLPClassifier, self).__init__()
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

def train_epoch(model, optimizer, criterion, train_dl):
    model.train()
    running_loss = 0.
    running_acc = 0.
    for x, y in train_dl:
        optimizer.zero_grad(set_to_none=True)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += (y_pred.argmax(dim=1) == y).mean().item()
    return running_loss / len(train_dl), running_acc / len(train_dl)

def test_epoch(model, criterion, test_dl):
    model.eval()
    running_loss = 0.
    running_acc = 0.
    with torch.no_grad():
        for x, y in test_dl:
            y_pred = model(x)
            loss = criterion(y_pred, y)

            running_loss += loss.item()
            running_acc += (y_pred.argmax(dim=1) == y).mean().item()
    return running_loss / len(test_dl), running_acc / len(test_dl)

def train(model, optimizer, criterion, scheduler, train_dl, val_dl, n_epochs):
    best_acc = 0
    best_model = None    
    for _ in range(n_epochs):
        loss_train, acc_train = train_epoch(model, optimizer, criterion, train_dl)
        loss_val, acc_val = test_epoch(model, criterion, val_dl)
        scheduler.step()
        print(f"Train loss: {loss_train:.4f}, Train acc: {acc_train:.4f}, Val loss: {loss_val:.4f}, Val acc: {acc_val:.4f}")
        
        if acc_val > best_acc:
            best_acc = acc_val
            best_model = model.state_dict()

    return best_model
