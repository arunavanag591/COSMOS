import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from figurefirst import FigureLayout,mpl_functions
import odor_stat_calculations as osc
from scipy.spatial.distance import cdist

## training
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler




class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device  # Add this line
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # 1 output (odor)

    def forward(self, x):
        if x.dim() == 3:  # If a batch of sequences is passed in
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        elif x.dim() == 2:  # If a single sequence is passed in
            h0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device)
            c0 = torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device)
        else:
            raise ValueError("Unexpected input dimension: %d" % x.dim())
        out, _ = self.lstm(x.unsqueeze(0) if x.dim() == 2 else x, (h0, c0))
        out = self.fc(out.squeeze(0) if x.dim() == 2 else out[:, -1, :])
        return out

    def reset_hidden_state(self, batch_size):
        self.hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
                       torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device))

        
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)  # Make LSTM bidirectional
        self.fc = nn.Linear(hidden_size * 2, 1)  # Adjust the input size of the fully connected layer
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        if x.dim() == 3:
            h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)  # Adjust the initial hidden state
            c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        elif x.dim() == 2:
            h0 = torch.zeros(self.num_layers * 2, 1, self.hidden_size).to(self.device)
            c0 = torch.zeros(self.num_layers * 2, 1, self.hidden_size).to(self.device)
        else:
            raise ValueError("Unexpected input dimension: %d" % x.dim())

        out, _ = self.lstm(x.unsqueeze(0) if x.dim() == 2 else x, (h0, c0))
        out = self.fc(out.squeeze(0) if x.dim() == 2 else out[:, -1, :])
        return out

    def reset_hidden_state(self, batch_size):
        self.hidden = (torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(self.device),
                       torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(self.device))
        
        
class CNNLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device):
        super(CNNLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.cnn = nn.Conv1d(input_size, hidden_size, kernel_size=1)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
#         x = x.view(batch_size * seq_length, 1, -1)
        x = x.view(batch_size, seq_length, -1).permute(0, 2, 1)

        x = self.cnn(x)
        x = x.view(batch_size, seq_length, -1)
        out, hidden = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out  # Return only the output


    def reset_hidden_state(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device))





def scale_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (((data - min_val) / (max_val - min_val)) * 10)

def load_and_preprocess_file(file_path, features, target):
    # Load the data
    df = pd.read_hdf(file_path)
    df['scaled_odor']=scale_data(df.odor)
    # Scale the features
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    # Convert data to PyTorch tensors
    feature_tensors = torch.Tensor(df[features].values)
    target_tensors = torch.Tensor(df[target].values)

    # Reshape to (seq_length, n_features)
    feature_tensors = feature_tensors.view(-1, len(features))

    return feature_tensors, target_tensors

def create_sequences(feature_tensors, target_tensors, seq_length):
    sequences = []
    targets = []
    for i in range(len(feature_tensors) - seq_length):
        sequences.append(feature_tensors[i:i+seq_length])
        targets.append(target_tensors[i+seq_length])
    return torch.stack(sequences), torch.stack(targets)


def train_on_single_file(model, optimizer, criterion, sequences, targets, num_epochs, batch_size):
    model.train()
    num_batches = len(sequences) // batch_size  # Determine the number of batches
    
    for epoch in range(num_epochs):
        for batch_idx in range(num_batches):  # Iterate over each batch
            # Get the current batch of sequences and targets
            batch_sequences = sequences[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            batch_targets = targets[batch_idx * batch_size : (batch_idx + 1) * batch_size]

            # Reset the hidden state for each new batch
            model.reset_hidden_state(batch_size)

            # Forward pass
            outputs = model(batch_sequences.view(-1, seq_length, input_size))
            # loss = criterion(outputs[:, -1, :], batch_targets.view(-1, 1))  # Compare only the last prediction with the target
            loss = criterion(outputs[:, -1], batch_targets.view(-1, 1))  # Compare only the last prediction with the target

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 50 ==0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
            
            
def whiff_mean_concentration(tensor):
    # print(torch.mean(tensor))
    return torch.mean(tensor)

## faster method but needs more gpu power
def train_on_single_file_faster(model, optimizer, criterion, sequences, targets, num_epochs):

    model.train()
    batch_size = sequences.size(0)
    for epoch in range(num_epochs):
        model.reset_hidden_state(batch_size)

        # Forward pass
        outputs = model(sequences.view(-1, seq_length, input_size))
        loss = criterion(outputs, targets.view(-1, 1))  # Compare the output directly with the target
         # Original MSE loss
        # mse_loss = criterion(outputs, targets.view(-1, 1))
        
        # # Additional 'whiff' loss
        # predicted_whiff_mean = whiff_mean_concentration(outputs)
        # actual_whiff_mean = whiff_mean_concentration(targets)
        # whiff_loss = torch.abs(predicted_whiff_mean - actual_whiff_mean)
        
        # # Combined loss
        # loss = mse_loss + whiff_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')




def predict_on_new_data(model, new_data_path, features, target, seq_length, device):
    # Load and preprocess the new data
    feature_tensors, target_tensors = load_and_preprocess_file(new_data_path, features, target)
    sequences, _ = create_sequences(feature_tensors, target_tensors, seq_length)

    # Move sequences to the device
    sequences = sequences.to(device)

    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(sequences.view(-1, seq_length, len(features)))

    # Convert the predictions to a numpy array
    predictions = predictions.cpu().numpy()

    return predictions



def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LSTM(input_size, hidden_size, num_layers, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.007)  # Define your optimizer
    criterion = nn.MSELoss()  # Define your loss function


    # Iterate over files
    filepath="/home/beast/An/data/train_new_axis/"
    for file in os.listdir(filepath):  # Replace with the actual path
        file_path = os.path.join(filepath, file)
        feature_tensors, target_tensors = load_and_preprocess_file(file_path, features, target)
        sequences, targets = create_sequences(feature_tensors, target_tensors, seq_length)
        sequences = sequences.to(device)
        targets = targets.to(device)
        train_on_single_file_faster(model, optimizer, criterion, sequences, targets, num_epochs)

    torch.save(model.state_dict(), '../assets/models/modelLSTMSept18-500epoch.pth')
    new_data_path = "/home/beast/An/data/train_new_axis/diag1.h5"
    predictions = predict_on_new_data(model, new_data_path, features, target, seq_length, device)
    test=pd.read_hdf(new_data_path)
    test['predicted_odor']=np.pad(predictions.flatten(),(0, len(test)-len(predictions)),mode='constant')

    f,ax=plt.subplots(1,1,figsize=(5,5))
    ax.plot(scale_data(test.odor), label='scaled measurements')
    ax.plot(test.predicted_odor, alpha=0.8,label='predictions')
    ax.set_ylabel('odor, v')
    ax.set_xlabel('samples')
    ax.legend()
    ax.set_title('200ep/timeseries')
    f.savefig('../assets/test.jpeg', dpi=150, bbox_inches = "tight")



features = ['distance_along_streakline','distance_from_source']
target = ['scaled_odor']
seq_length = 5  # Choose a suitable sequence length
num_epochs = 500  # Choose a suitable number of epochs

# Initialize LSTM model
input_size = 2  # Number of features
hidden_size = 128
num_layers = 4

if __name__ == "__main__":
    main()
