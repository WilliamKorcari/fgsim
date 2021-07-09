from comet_ml import Experiment

# Import comet_ml at the top of your file
# Create an experiment with your api key
experiment = Experiment(
    api_key='',
    project_name="general",
    workspace="",
)


import os
import time
import numpy as np
from HGCalShowers import HGCalShowers
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool, BatchNorm
from utils import EarlyStopping
from utils import sumInputs
hyperp = {
    'epochs': 300,
    'h_channels': 64,
    'learning_rate': 0.01, 
    'batch_size': 64,
    'patience': 30,
    'min_delta': 0.0001,
    'save_path': '/beegfs/desy/user/korcariw/hgcal_model/trained_models/fixed_test_bnorm_input_globpool_model.pt',   
    'hidden_d_layers': 2,
    'gcn_layers': 2,
}


experiment.log_parameters(hyperp)


fullDataLoad = """showers = {}
for i in range(18):
    showers[i] = HGCalShowers(root = '/beegfs/desy/user/korcariw/hgcal_model/', 
                       raw_files = [f'ntupleTree_{i}.root'],             
                       out_file = f'photonShowers50_100GeV_{i}.pt', 
                       include_labels = True,
                       load_on_gpu = True
                      )


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Model will be loaded on {device}..')
train_dataset = showers[0]
for i in range(1, 14):
    train_dataset += showers[i]
test_dataset = showers[15]
for i in range(15, 18):
    test_dataset += showers[i]
print('Splitting the data..')
"""


#Data preparation PARTIAL

showers = HGCalShowers(root = '/beegfs/desy/user/korcariw/hgcal_model/', 
                       raw_files = ['ntupleTree_0.root'],             
                       out_file = 'photonShowers50_100GeV_0.pt', 
                       include_labels = True,
                       load_on_gpu = True
                      )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Model will be loaded on {device}..')
print('Splitting the data.. ')
train_dataset = showers[:180000]
test_dataset = showers[180000:190000]
val_dataset = showers[190000:]

train_loader = DataLoader(train_dataset, batch_size=hyperp['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=hyperp['batch_size'], shuffle=False) 

print("Done.")



class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(126755)
        
        self.bnorm = BatchNorm(4)
        self.conv0 = GCNConv(4, hidden_channels)
        self.convs  =torch.nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(1, hyperp['gcn_layers'])])
        self.bnorm1 = BatchNorm(hidden_channels)
        self.lins   = torch.nn.ModuleList([Linear(hidden_channels, hidden_channels) for _ in range(hyperp['hidden_d_layers'])])
        self.out   = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.bnorm(x)
        x = self.conv0(x, edge_index)
        for gc in self.convs:
            x = gc(x, edge_index)
        # 2. Readout layer
        x = global_add_pool(x, batch)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = self.bnorm1(x)
        for l in self.lins:
            x = l(x)
        x = self.out(x)
        return x

print('Building model..')
model = GCN(hidden_channels=hyperp['h_channels'])
model.to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=hyperp['learning_rate'])
criterion = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose = True)
early_stopping = EarlyStopping(path = hyperp['save_path'], patience = hyperp['patience'], min_delta = hyperp['min_delta'])
print("Done.")

def train():
    model.train()
    training_loss = 0
    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))  # Perform a single forward pass.
        loss = criterion(out.view((len(out))), data.y.to(device))  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        training_loss += loss.item()

def test(loader):
    model.eval()
    loss = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
        pred = criterion(out.view((len(out))), data.y.to(device))
        loss += pred.item()
    return pred  # Derive ratio of correct predictions.

print('Starting train of the model..\n')
with experiment.train():
    for epoch in range(hyperp['epochs']):
        start = time.time()
        train()
        train_mse = test(train_loader)
        val_mse = test(val_loader)
        scheduler.step(val_mse)
        experiment.log_metric("train_MSE", train_mse, step=epoch)
        experiment.log_metric("val_MSE", val_mse, step=epoch)
        ellapsed_time = time.time() - start
        print(f'Epoch: {epoch:03d}, Time: {ellapsed_time:.1f}s, Train MSE: {train_mse:.4f}, Test MSE: {val_mse:.4f}')
        early_stopping(model.state_dict(), val_mse)
        if early_stopping.early_stop:
            break
model.load_state_dict(torch.load(hyperp['save_path']))

with experiment.test():
    with torch.no_grad():
        model.eval()
        #for data in test_loader:
        out = torch.tensor([test_model(data.x.to(device), data.edge_index.to(device), data.batch.to(device)) for data in test_loader])
        test_mse = criterion(out.view((len(out))), data.y.to(device))
        experiment.log_metric("test_MSE", test_mse)

