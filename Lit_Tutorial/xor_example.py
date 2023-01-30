import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm



torch.manual_seed(2)


class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x 

act_fn_dict = {
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid
}

class XOR(nn.Module):
    def __init__(
        self,
        in_features : int = 2,
        out_features : int = 1,
        hidden_dim : int = 4,
        act_fn : str = 'tanh'
    ):
        super().__init__()
        self.f1 = nn.Linear(in_features, hidden_dim)
        self.act_fn = act_fn_dict[act_fn]()
        self.f2 = nn.Linear(hidden_dim, out_features)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.f1(x)
        x = self.act_fn(x)
        x = self.f2(x)
        return x  


class MyModule2(nn.Module):
    def __init__(self):
        super().__init__()
        self.choices = nn.ModuleDict({
                'conv': nn.Conv2d(10, 10, 3),
                'pool': nn.MaxPool2d(3)
        })
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['prelu', nn.PReLU()]
        ])

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x
        

net = XOR()
# # print(net)
# for n, p in net.named_parameters():
#     print(n, p.shape)

class XORDataset(Dataset):
    def __init__(
        self,
        size : int = 100,
        std : float = 0.1
        ):
        super().__init__()
        self.size = size
        self.std = std
        self.generate_continuous_xor()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data = self.data[idx]    
        label = self.label[idx]

        return data, label

    def generate_continuous_xor(self):
        data = torch.randint(low=0, high=2, size=(self.size, 2), dtype=torch.float32)
        label = ((data.sum(-1) == 1) * 1.)

        data += self.std * torch.randn(data.shape)

        self.data = data
        self.label = label


data = XORDataset(size=1000)
# print(data.label)

dataloader = DataLoader(data, shuffle=True, batch_size=16)
# print(iter(dataloader))


optimizer = optim.SGD(net.parameters(), lr=0.1)
loss_module = nn.BCEWithLogitsLoss()



def train_model(model, optimizer, dataloader, loss_module, num_epochs=50):
    model.train()

    for epoch in range(num_epochs):
        for x_batch, y_batch in dataloader:
            x_batch = x_batch
            y_batch = y_batch

            out = model(x_batch)
            out = torch.squeeze(out)
            loss = loss_module(out, y_batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch}", loss.item())  

          



# train_model(net, optimizer, dataloader, loss_module)
# torch.save(net.state_dict(), 'xor_model.tar')

# load save model weights
state_dict = torch.load("xor_model.tar")
new_model = XOR()
new_model.load_state_dict(state_dict)


test_dataset = XORDataset(size=500)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=16)


new_model.eval()
accuracy = []
for x,y in test_dataloader:
    with torch.no_grad():
        out = new_model(x)
        out = out.squeeze()
        out = (torch.sigmoid(out) > 0.5) * 1.
        
        accuracy.append(torch.mean((out == y)*1., axis=-1).item())

mean_accuracy = sum(accuracy) / len(accuracy)       
print("Accuracy", mean_accuracy)




