import torch
from torch import nn
from itertools import starmap
import yaml
from yaml.loader import SafeLoader

dims = '../config/models_dims.yaml'
params = '../config/params.yaml'

with open(dims) as f:
    config = yaml.load(f, Loader=SafeLoader)

    # batch norm and dropout parameters
    EPS = config['eps']
    MOM = config['mom']
    DROP = config['drop']

    # CNN parameters
    PAD = config['padding']
    STRD = config['stride']
    CHS = config['channels']
    KER = config['kernels']

    # LSTM parameters
    HDS = config['hidden_layer_size']
    NUML = config['num_cells']

    # MLP parameters
    HID = config['hidden_layers']
    WID = config['width']

with open(params) as f:
    config = yaml.load(f, Loader=SafeLoader)

    # input size
    SIZE = config['size']


### CNN model

class Block(nn.Module):
    def __init__(self, 
                 in_ch, 
                 out_ch,
                 ker):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch,
                               out_ch,
                               kernel_size=ker,
                               stride=STRD,
                               padding=PAD)
        
        self.batch_norm = nn.BatchNorm1d(num_features=out_ch,
                                         eps=EPS, 
                                         momentum=MOM)
        self.relu  = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        return self.relu(x)
    
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = starmap(Block, zip(CHS, CHS[1:], KER))
        self.enc_blocks = nn.ModuleList(self.l)
        self.pool       = nn.AdaptiveMaxPool1d(1)
        self.relu  = nn.ReLU()
            
    def forward(self, x):
            for block in self.enc_blocks:
                x = block(x)
                x = self.pool(x)
            return x
    
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.linear_relu_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(CHS[-1], SIZE),
            nn.ELU(),
            nn.Dropout(DROP),
            nn.Linear(SIZE, SIZE),
        )
    
    def forward(self, x):
        x = self.encoder(x.unsqueeze(1))
        x = self.linear_relu_stack(x)
        return x.squeeze().sigmoid()

#### MLP baseline

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        layer = (nn.Linear(WID, WID), nn.ReLU())
        self.layers = layer * HID
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(SIZE, WID),
            nn.BatchNorm1d(num_features=WID,
                                         eps=EPS, 
                                         momentum=MOM),
            nn.ReLU(),
            *self.layers,
            nn.Dropout(DROP),
            nn.Linear(WID, SIZE),
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x.sigmoid()
   
### Bidirectional LSTM
 
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.hidden_layer_size = HDS
        self.lstm = nn.LSTM(SIZE, 
                            hidden_size=HDS,
                            num_layers=NUML,
                            batch_first=True, 
                            bidirectional=True)
        self.batchnorm = nn.BatchNorm1d(num_features=HDS,
                                         eps=EPS, 
                                         momentum=MOM)
        self.linear = nn.Linear(2*HDS, SIZE)
        self.dropout = nn.Dropout(DROP)

    def forward(self, x):
        lstm_out, self.hidden_cell = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        y = self.linear(lstm_out.squeeze())
        return y.sigmoid()
    
### LSTM-FCN

class BlockLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=SIZE, 
                            hidden_size=HDS, 
                            bidirectional=True)

    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        x, self.hidden_cell = self.lstm(x)
        return x

class LSTMFCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_block = BlockLSTM()
        self.fcn_block = Encoder()
        self.softmax = nn.Softmax()
        self.linear = nn.Linear(2*HDS + CHS[-1], SIZE)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x1 = self.lstm_block(x).squeeze()
        x2 = self.fcn_block(x).squeeze()
        x = torch.cat([x1, x2], 1)
        y = self.linear(x)
        return y.sigmoid()
