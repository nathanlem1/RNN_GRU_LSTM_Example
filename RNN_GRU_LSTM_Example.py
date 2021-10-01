"""
This gives an example of using RNN, GRU and LSTM recurrent architectures in PyTorch. When compared to the vanilla RNN,
GRU has two gates: update gate and reset (relevance) gate, and LSTM has three gates: input (update) gate, forget gate
and output gate.  These 3 recurrent architectures are implemented in such as way that one can use one of them at a
time as well as learn what their relations and differences are when using them.
Refer to https://pytorch.org/docs/stable/nn.html#rnn for PyTorch documentation to learn more.

Important hyper-parameters you can play with:
a) num_layers - you can change this e.g. 1, 2, 3, 4, ...
b) num_directions - 1 for Unidirectional (forward directional only) RNN/GRU/LSTM   OR  2 for Bidirectional RNN/GRU/LSTM.

"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(1)    # Reproducible

# Some important hyper-parameters
seq_length = 10  # Sequence length
input_size = 1  # Input size
batch = 1  # Batch size
hidden_size = 32  # The number of features in the hidden state h
num_layers = 2  # 2 # Number of RNN/GRU/LSTM layers E.g., setting num_layers=2 would mean stacking two RNNs/GRUs/LSTMs
# together to form a stacked RNN/GRU/LSTM, with the second RNN/GRU/LSTM taking in outputs of the first RNN/GRU/LSTM
# and computing the final results.

num_directions = 2  # 2 # Unidirectional (forward directional only) RNN/GRU/LSTM; if Bidirectional RNN/GRU/LSTM,
# num_directions = 2

num_epochs = 100  # Number of epochs
output_dim = 1  # Output dimension
LR = 0.02  # Learning rate

# Decide whether to use unidirectional or bidirectional recurrent network
if num_directions == 1:
    bidirectional = False
elif num_directions == 2:
    bidirectional = True
else:
    print('Usage: 1 for unidirectional or 2 for bidirectional. ')
    Exception('Wrong num_directions')


class RNN_GRU_LSTM(nn.Module):
    def __init__(self, flag):
        super(RNN_GRU_LSTM, self).__init__()
        self.flag = flag

        if self.flag == 'RNN':
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif self.flag  == 'GRU':
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif self.flag  == 'LSTM':
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,  # If batch_first=True, input & output will have batch size as 1st dimension. e.g.
                # (batch, seq_length, input_size); default is False with (seq_length, batch, input_size) format

                bidirectional=bidirectional,
            )
        else:
            print('Usage: Set to flag to RNN or LSTM or GRU. ')
            Exception('Wrong recurrent architecture of choice')

        self.out = nn.Linear(hidden_size*num_directions, output_dim)

    def forward(self, x, h_state, c_state):
        # x (seq_length, batch, input_size) or (batch, seq_length, input_size) if batch_first=True
        # h_state (num_layers*num_directions, batch, hidden_size)
        # r_out (seq_len, batch, hidden_size) or (batch, seq_len, hidden_size) if batch_first=True
        if self.flag  == 'RNN':
            r_out, h_state = self.rnn(x, h_state)
            c_state = None                    # No cell state output from RNN
        elif self.flag == 'GRU':
            r_out, h_state = self.gru(x, h_state)
            c_state = None                  # No cell state output from GRU
        else:  # self.flag == LSTM
            r_out, (h_state, c_state) = self.lstm(x, (h_state, c_state))

        # nn.Linear can accept inputs of any dimension and returns outputs with same dimension except for the last
        outs = self.out(r_out)
        return outs, h_state, c_state


# Set a flag to choose one of RNN, GRU or LSTM
flag = 'LSTM'

# Initialize the model
model = RNN_GRU_LSTM(flag=flag)
print(model)

# Set the Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Set to Mean Square Error (MSE) loss
loss = nn.MSELoss()

# Initialize hidden state h_0 and cell state c_0
h_state = torch.zeros(num_layers*num_directions, batch, hidden_size)  # For initial hidden state, h_0.
c_state = torch.zeros(num_layers*num_directions, batch, hidden_size)  # For initial cell state, c_0. NOTE: If (h_0, c_0)
# is not provided, both h_0 and c_0 default to zero.

plt.figure(1, figsize=(12, 5))
plt.ion()  # To continuously plot

hist = np.zeros(num_epochs)  # Keep history of losses for plotting purpose

for step in range(num_epochs):

    # Use sin to predict to cos
    start, end = step * np.pi, (step + 1) * np.pi  # Set a time range
    steps = np.linspace(start, end, seq_length, dtype=np.float32,  endpoint=False)  # float32 for converting torch
    # FloatTensor

    x_np = np.sin(steps)
    y_np = np.cos(steps)

    # x = torch.from_numpy(x_np[:, np.newaxis, np.newaxis])  # shape (seq_length, batch, input_size). Use this if
    # # batch_first is not set to True
    #
    # y = torch.from_numpy(y_np[:, np.newaxis, np.newaxis])

    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])  # shape (batch, seq_length, input_size)
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    # Forward propagation
    if flag == 'RNN':
        prediction, _, h_state = model(x, None, h_state)  # RNN output
    elif flag == 'GRU':
        prediction, _, h_state = model(x, None, h_state)  # GRU output
    else:
        prediction, h_state, c_state = model(x, h_state, c_state)  # LSTM output
        c_state = c_state.data  # Repack the cell state, break the connection from last iteration.
        # This is very important!

    h_state = h_state.data  # Repack the hidden state, break the connection from last iteration. This is very important!

    # Compute loss
    loss_res = loss(prediction, y)

    # Clear gradients for each training step, otherwise they will accumulate between epochs
    optimizer.zero_grad()

    # Backpropagation, compute gradients
    loss_res.backward()

    # Update parameters; apply gradients
    optimizer.step()

    # Print loss values at some epochs
    if step % 10 == 0:
        print('Epoch {}, MSE: {:.4f}'.format(step, loss_res.item()))
    hist[step] = loss_res.item()

    # Plotting
    plt.plot(steps, y_np.flatten(), 'r-')  # Ground truth
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')  # Prediction
    plt.draw()
    plt.pause(0.05)

plt.ioff()
plt.show()

plt.plot(hist, label="Training loss")
plt.legend()
plt.show()
