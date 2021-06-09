class MyModel_1(nn.Module):
    def __init__(self, 
                 input_dim, 
                 embedding_dim, 
                 hidden_dim, 
                 output_dim1,
                 output_dim2,
                 n_layers, 
                 bidirectional, 
                 dropout, 
                 pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)
        
        
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers = n_layers, 
                            bidirectional = bidirectional,
                            dropout = dropout if n_layers > 1 else 0)
        self.T1 = torch.nn.Transformer(d_model=hidden_dim)
        self.T2 = torch.nn.Transformer(d_model=hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim, output_dim1)
        self.fc2 = nn.Linear(hidden_dim, output_dim2)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, y1, y2):

        #text = [sent len, batch size]
        
        #pass text through embedding layer
        embedded_X = self.dropout(self.embedding(text))
        embedded_y1 = self.dropout(self.embedding(y1))
        embedded_y2 = self.dropout(self.embedding(y2))
        
        #embedded = [sent len, batch size, emb dim]
        #pass embeddings into LSTM
        shared_output, (hidden, cell) = self.lstm(embedded_X)
#         print(shared_output.shape)
        #outputs holds the backward and forward hidden states in the final layer
        #hidden and cell are the backward and forward hidden and cell states at the final time-step
        
        #output = [sent len, batch size, hid dim * n directions]
        #hidden/cell = [n layers * n directions, batch size, hid dim]
        
        #we use our outputs to make a prediction of what the tag should be
        out1 = self.T1(shared_output, embedded_y1)
        out2 = self.T2(shared_output, embedded_y2)
        predictions_1 = self.fc1(self.dropout(out1))
        predictions_2 = self.fc2(self.dropout(out2))
        
        #predictions = [sent len, batch size, output dim]
        
        return predictions_1, predictions_2


class MyModel_3(nn.Module):
    def __init__(self, 
                 input_dim, 
                 embedding_dim, 
                 hidden_dim, 
                 output_dim1,
                 output_dim2,
                 n_layers, 
                 bidirectional, 
                 dropout, 
                 pad_idx, input_dimA, input_dimB):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)
        self.embeddingA = nn.Embedding(input_dimA, embedding_dim, padding_idx = pad_idx)
        self.embeddingB = nn.Embedding(input_dimB, embedding_dim, padding_idx = pad_idx)
        
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers = n_layers, 
                            bidirectional = bidirectional,
                            dropout = dropout if n_layers > 1 else 0)
        self.T1 = torch.nn.Transformer(d_model=hidden_dim)
        self.T2 = torch.nn.Transformer(d_model=hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim, output_dim1)
        self.fc2 = nn.Linear(hidden_dim, output_dim2)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, y1, y2):

        #text = [sent len, batch size]
        
        #pass text through embedding layer
        embedded_X = self.dropout(self.embedding(text))
        embedded_y1 = self.dropout(self.embeddingA(y1))
        embedded_y2 = self.dropout(self.embeddingB(y2))
        
        #embedded = [sent len, batch size, emb dim]
        #pass embeddings into LSTM
        shared_output, (hidden, cell) = self.lstm(embedded_X)
#         print(shared_output.shape)
        #outputs holds the backward and forward hidden states in the final layer
        #hidden and cell are the backward and forward hidden and cell states at the final time-step
        
        #output = [sent len, batch size, hid dim * n directions]
        #hidden/cell = [n layers * n directions, batch size, hid dim]
        
        #we use our outputs to make a prediction of what the tag should be
        out1 = self.T1(shared_output, embedded_y1)
        out2 = self.T2(shared_output, embedded_y2)
        predictions_1 = self.fc1(self.dropout(out1))
        predictions_2 = self.fc2(self.dropout(out2))
        
        #predictions = [sent len, batch size, output dim]
        
        return predictions_1, predictions_2