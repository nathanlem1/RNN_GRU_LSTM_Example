## RNN_GRU_LSTM_Example

This gives an example of using RNN, GRU and LSTM recurrent architectures in PyTorch. 
When compared to the vanilla RNN, GRU has two gates: update gate and reset (relevance) 
gate, and LSTM has three gates: input (update) gate, forget gate and output gate. 
These 3 recurrent architectures are implemented in this example in such as way that one 
can use one of them at a time as well as learn what their relations and differences are 
when using them. Both Unidirectional (forward directional) and bidirectional RNN, GRU and 
LSTM are included. Refer to [RNN](https://pytorch.org/docs/stable/nn.html#rnn) for PyTorch documentation to learn more.

Important hyper-parameters you can play with: \
a) num_layers - you can change this e.g. 1, 2, 3, 4, ... \
b) num_directions - 1 for Unidirectional (forward directional only) RNN/GRU/LSTM   or
   2 for Bidirectional RNN/GRU/LSTM.
   
 <br/>

## Getting Started
```bash
$ git clone https://github.com/nathanlem1/RNN_GRU_LSTM_Example.git
$ cd RNN_GRU_LSTM_Example
$ python SVM_PyTorch.py
```

<br/>

## Dependencies
* [Python 3.5+](https://www.python.org/downloads/)
* [PyTorch 0.4.0+](http://pytorch.org/)


