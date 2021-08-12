import torch
import torch.nn as nn
import torch_geometric.nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from torch_geometric_temporal.nn.recurrent import GConvLSTM

class MTLRecurrentGCN(nn.Module):
    """
    Multi-task learning recurrent GCN
    """
    def __init__(self, node_features, leadtime_range):
        super().__init__()

        self.recurrent = GConvLSTM(node_features, 32, 3)
        self.fc1 = nn.Linear(32, 16)

        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)

        self.bin_fc1 = nn.Linear(8, 1)

        self.lead_fc1 = nn.Linear(8, leadtime_range+1)

        self.dropout = nn.Dropout()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, batch, src, dst):
      
        h_0 = None
        c_0 = None
        
        # batch is a list of Batch objects
        for b in batch:
            h_0, c_0 = self.recurrent(X = x, edge_index = b.edge_index, H = h_0, C = c_0)

        h = F.relu(h_0)

        h = self.fc1(h)
        h = F.relu(h)
        
        h = self.fc2(torch.cat((h[src], h[dst]), axis = 1))
        h = F.relu(h)

        h = self.dropout(h)
        h = self.fc3(h)
        h = F.relu(h)

        # non MTL model would either remove bin_h or lead_h
        bin_h = self.dropout(h)
        bin_h = self.bin_fc1(bin_h)
        bin_h = torch.sigmoid(bin_h)
        
        lead_h = self.dropout(h)
        lead_h = self.lead_fc1(lead_h)
        lead_h = self.softmax(lead_h)

        return bin_h, lead_h

class BinRecurrentGCN(nn.Module):
    """
    Binary classification recurrent GCN
    """
    def __init__(self, node_features):
        super().__init__()

        self.recurrent = GConvLSTM(node_features, 32, 3)
        self.fc1 = nn.Linear(32, 16)

        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)

        self.bin_fc1 = nn.Linear(8, 1)

        self.dropout = nn.Dropout()

    def forward(self, x, batch, src, dst):
      
        h_0 = None
        c_0 = None
        # edge_indices is a list of Batch objects
        for b in batch:
            h_0, c_0 = self.recurrent(X = x, edge_index = b.edge_index, H = h_0, C = c_0)

        h = F.relu(h_0)

        h = self.fc1(h)
        h = F.relu(h)
        
        h = self.fc2(torch.cat((h[src], h[dst]), axis = 1))
        h = F.relu(h)

        h = self.dropout(h)
        h = self.fc3(h)
        h = F.relu(h)

        bin_h = self.dropout(h)
        bin_h = self.bin_fc1(bin_h)
        bin_h = torch.sigmoid(bin_h)

        return bin_h

class LeadRecurrentGCN(nn.Module):
    """
    Leadtime recurrent GCN
    """
    def __init__(self, node_features, leadtime_range):
        super().__init__()

        self.recurrent = GConvLSTM(node_features, 32, 3)
        self.fc1 = nn.Linear(32, 16)

        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)

        self.lead_fc1 = nn.Linear(8, leadtime_range+1)

        self.dropout = nn.Dropout()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, batch, src, dst):
      
        h_0 = None
        c_0 = None
        # edge_indices is a list of Batch objects
        for b in batch:
            h_0, c_0 = self.recurrent(X = x, edge_index = b.edge_index, H = h_0, C = c_0)

        h = F.relu(h_0)

        h = self.fc1(h)
        h = F.relu(h)
        
        h = self.fc2(torch.cat((h[src], h[dst]), axis = 1))
        h = F.relu(h)

        h = self.dropout(h)
        h = self.fc3(h)
        h = F.relu(h)

        lead_h = self.dropout(h)
        lead_h = self.lead_fc1(lead_h)
        lead_h = self.softmax(lead_h)

        return lead_h

class MTLGCN(nn.Module):
    """
    Multi-task learning GCN
    """
    def __init__(self, node_features, leadtime_range):
        super().__init__()

        self.gcn1 = ChebConv(node_features, 32, 3)
        
        self.fc1 = nn.Linear(32, 16)

        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)

        self.bin_fc1 = nn.Linear(8, 1)

        self.lead_fc1 = nn.Linear(8, leadtime_range+1)

        self.dropout = nn.Dropout()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, batch, src, dst):
      
        # batch is a Batch object 
        h = self.gcn1(x = x, edge_index = batch.edge_index)
        h = torch.tanh(h)

        h = self.fc1(h)
        h = F.relu(h)
        
        h = self.fc2(torch.cat((h[src], h[dst]), axis = 1))
        h = F.relu(h)

        h = self.dropout(h)
        h = self.fc3(h)
        h = F.relu(h)

        # non MTL model would either remove bin_h or lead_h
        bin_h = self.dropout(h)
        bin_h = self.bin_fc1(bin_h)
        bin_h = torch.sigmoid(bin_h)
        
        lead_h = self.dropout(h)
        lead_h = self.lead_fc1(lead_h)
        lead_h = self.softmax(lead_h)

        return bin_h, lead_h

class BinGCN(nn.Module):
    """
    Binary classification GCN
    """
    def __init__(self, node_features):
        super().__init__()

        self.gcn1 = ChebConv(node_features, 32, 3)
        
        self.fc1 = nn.Linear(32, 16)

        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)

        self.bin_fc1 = nn.Linear(8, 1)

        self.dropout = nn.Dropout()

    def forward(self, x, batch, src, dst):
      
        # batch is a Batch object 
        h = self.gcn1(x = x, edge_index = batch.edge_index)
        h = torch.tanh(h)

        h = self.fc1(h)
        h = F.relu(h)
        
        h = self.fc2(torch.cat((h[src], h[dst]), axis = 1))
        h = F.relu(h)

        h = self.dropout(h)
        h = self.fc3(h)
        h = F.relu(h)

        bin_h = self.dropout(h)
        bin_h = self.bin_fc1(bin_h)
        bin_h = torch.sigmoid(bin_h)

        return bin_h

class LeadGCN(nn.Module):
    """
    Leadtime GCN
    """
    def __init__(self, node_features, leadtime_range):
        super().__init__()

        self.gcn1 = ChebConv(node_features, 32, 3)
        
        self.fc1 = nn.Linear(32, 16)

        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)


        self.lead_fc1 = nn.Linear(8, leadtime_range+1)

        self.dropout = nn.Dropout()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, batch, src, dst):
      
        # batch is a Batch object 
        h = self.gcn1(x = x, edge_index = batch.edge_index)
        h = torch.tanh(h)

        h = self.fc1(h)
        h = F.relu(h)
        
        h = self.fc2(torch.cat((h[src], h[dst]), axis = 1))
        h = F.relu(h)

        h = self.dropout(h)
        h = self.fc3(h)
        h = F.relu(h)
        
        lead_h = self.dropout(h)
        lead_h = self.lead_fc1(lead_h)
        lead_h = self.softmax(lead_h)

        return lead_h
