import torch
import torch.nn as nn
import numpy as np
import random
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
from tqdm import tqdm

from utils import categorical_accuracy

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

def train_model(model, iterator, optimizer, criterion, tag_pad_idx):
    epoch_loss = 0
    epoch_acc_pos = 0
    epoch_acc_neg = 0
    model.train()
    for batch in tqdm(iterator):
        text = batch.Sentence
        pos = batch.POS
        neg_scope = batch.Neg_Scope
        optimizer.zero_grad()
        #text = [sent len, batch size]
        predictions1, predictions2 = model(text, pos[:-1,:], neg_scope[:-1,:])
        
        #predictions = [sent len, batch size, output dim]
        #tags = [sent len, batch size]
        
        predictions1 = predictions1.view(-1, predictions1.shape[-1])
        predictions2 = predictions2.view(-1, predictions2.shape[-1])
        pos = pos[1:,:].view(-1)
        neg_scope = neg_scope[1:,:].view(-1)
        
        #predictions = [sent len * batch size, output dim]
        #tags = [sent len * batch size]
        
        loss1 = criterion(predictions1, pos) 
        loss2 = criterion(predictions2, neg_scope)
        
        loss = loss1+loss2
                
        acc_pos = categorical_accuracy(predictions1, pos, tag_pad_idx)
        acc_neg = categorical_accuracy(predictions2, neg_scope, tag_pad_idx)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc_pos += acc_pos.item()
        epoch_acc_neg += acc_neg.item()
        
    return epoch_loss / len(iterator), epoch_acc_pos / len(iterator), epoch_acc_neg / len(iterator)

def evaluate(model, iterator, criterion, tag_pad_idx):
    
    epoch_loss = 0
    epoch_acc_pos = 0
    epoch_acc_neg = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            text = batch.Sentence
            pos = batch.POS
            neg_scope = batch.Neg_Scope
            
            predictions1, predictions2 = model(text, pos[:-1,:], neg_scope[:-1,:])
            
            predictions1 = predictions1.view(-1, predictions1.shape[-1])
            predictions2 = predictions2.view(-1, predictions2.shape[-1])
            pos = pos[1:,:].view(-1)
            neg_scope = neg_scope[1:,:].view(-1)
            
            loss1 = criterion(predictions1, pos) 
            loss2 = criterion(predictions2, neg_scope)

            loss = loss1+loss2

            acc_pos = categorical_accuracy(predictions1, pos, tag_pad_idx)
            acc_neg = categorical_accuracy(predictions2, neg_scope, tag_pad_idx)

            epoch_loss += loss.item()
            epoch_acc_pos += acc_pos.item()
            epoch_acc_neg += acc_neg.item()
        
    return epoch_loss / len(iterator), epoch_acc_pos / len(iterator), epoch_acc_neg / len(iterator)

class MyModel_2(nn.Module):
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
        self.T1 = torch.nn.Transformer(d_model=hidden_dim,num_encoder_layers=3, num_decoder_layers=1)
        self.T2 = torch.nn.Transformer(d_model=hidden_dim,num_encoder_layers=3, num_decoder_layers=1)
        
        self.fc1 = nn.Linear(hidden_dim, output_dim1)
        self.fc2 = nn.Linear(hidden_dim, output_dim2)
        
        self.dropout = nn.Dropout(dropout)
    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)

    def forward(self, text, y1, y2):

        #text = [sent len, batch size]
        
        #pass text through embedding layer
        src_pad_mask = self.make_len_mask(text)
        trg_pad_mask1 = self.make_len_mask(y1)
        trg_pad_mask2 = self.make_len_mask(y2)
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
        src_mask = self.T1.generate_square_subsequent_mask(len(shared_output)).to(shared_output.device)
        trg_mask1 = self.T1.generate_square_subsequent_mask(len(embedded_y1)).to(embedded_y1.device)
        trg_mask2 = self.T2.generate_square_subsequent_mask(len(embedded_y2)).to(embedded_y2.device)
        out1 = self.T1(shared_output, embedded_y1, tgt_mask=trg_mask1,src_mask=src_mask,src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask1, memory_key_padding_mask=src_pad_mask)
        out2 = self.T2(shared_output, embedded_y2, tgt_mask=trg_mask2,src_mask=src_mask,src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask2, memory_key_padding_mask=src_pad_mask)
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


# import math
# from einops import rearrange

# import torch
# from torch import nn


# class LanguageTransformer(nn.Module):
#     def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, pos_dropout, trans_dropout):
#         """
#         Initializes the model
#                 Parameters:
#                         vocab_size (int): The amount of tokens in both vocabularies (including start, end, etc tokens)
#                         d_model (int): Expected number of features in the encoder/decoder inputs, also used in embeddings
#                         nhead (int): Number of heads in the transformer
#                         num_encoder_layers (int): Number of sub-encoder layers in the transformer
#                         num_decoder_layers (int): Number of sub-decoder layers in the transformer
#                         dim_feedforward (int): Dimension of the feedforward network in the transformer
#                         max_seq_length (int): Maximum length of each tokenized sentence
#                         pos_dropout (float): Dropout value in the positional encoding
#                         trans_dropout (float): Dropout value in the transformer
#         """
#         super().__init__()
#         self.d_model = d_model
#         self.embed_src = nn.Embedding(vocab_size, d_model)
#         self.embed_tgt = nn.Embedding(vocab_size, d_model)
#         self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)

#         self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, trans_dropout)
#         self.fc1 = nn.Linear(d_model, vocab_size)
#         self.fc2 = nn.Linear(d_model, vocab_size)

#     def forward(self, src, tgt, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask, tgt_mask):
#         # Reverse the shape of the batches from (num_sentences, num_tokens_in_each_sentence)
#         src = rearrange(src, 'n s -> s n')
#         tgt = rearrange(tgt, 'n t -> t n')

#         # Embed the batches, scale by sqrt(d_model), and add the positional encoding
#         src = self.pos_enc(self.embed_src(src) * math.sqrt(self.d_model))
#         tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))

#         # Send the batches to the model
#         output = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask,
#                                   tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

#         # Rearrange to batch-first
#         output = rearrange(output, 't n e -> n t e')

#         # Run the output through an fc layer to return values for each token in the vocab
#         return self.fc(output)


# # Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=100):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)