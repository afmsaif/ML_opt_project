import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NN(nn.Module):
    def __init__(self,NFILT=256,NOUT=4):
        super(NN,self).__init__()
        self.conv0 = nn.Conv2d(1,NFILT,kernel_size=(200,3),padding=(0,1),bias=False)
        self.bn0 = nn.BatchNorm2d(NFILT)
        self.gru = nn.GRU(input_size=NFILT,hidden_size=128,num_layers=1,batch_first=True,bidirectional=False)
        self.fc1 = nn.Linear(128,NOUT)



    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = x.squeeze().permute(0,2,1)
        x,_ = self.gru(x)
        x = F.dropout(x,p=0.5,training=self.training)
        x = self.fc1(x)
        return x



class CNNLSTMModel(nn.Module):
    def __init__(self, num_classes=4):
        super(CNNLSTMModel, self).__init__()
        
        # CNN Encoder
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d((2, 2))
        
        # Calculate flattened size for fully connected layer
        sample_input = torch.rand(1, 1, 200, 116)  # Dummy input to calculate flattened size
        with torch.no_grad():
            flattened_size = self._get_flattened_size(sample_input)

        # Fully connected layer after CNN
        self.fc = nn.Linear(flattened_size, 128)
        
        # LSTM Decoder
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.output_fc = nn.Linear(64, num_classes)

    def _get_flattened_size(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        return x.view(1, -1).size(1)

    def forward(self, x):
        # CNN Encoder forward pass
        batch_size, _, _, _ = x.size()  # Assuming input shape (batch, channel, time, features)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        
        # Flatten and fully connected layer
        x = x.view(batch_size, -1)  # Flatten
        x = self.fc(x)
        
        # Add sequence dimension for LSTM
        x = x.unsqueeze(1)  # Shape becomes (batch, seq_len=1, features=128)
        
        # LSTM Decoder forward pass
        x, _ = self.lstm(x)
        x = self.output_fc(x[:, -1, :])  # Use the last LSTM output for classification
        
        return x

