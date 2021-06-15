import torch
import torch.nn as nn
import numpy as np
import random
SEED = 42
import math

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
from tqdm import tqdm

from utils import categorical_accuracy
# from crf_layer import CRF
from torchcrf import CRF

def print_graph(g, level=0):
    if g == None: return
    print(level)
    for subg in g.next_functions:
        print_graph(subg[0], level+1)

# Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=85):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def train_model(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc_pos = 0
    epoch_acc_neg = 0
    model.train()
    for batch in tqdm(iterator):
        text = batch.Sentence
        pos = batch.POS
        neg_scope = batch.Neg_Scope
        optimizer.zero_grad()
        predictions1, predictions2 = model(text, pos[:-1,:], neg_scope[:-1,:])
        loss1 = -model.crf1(predictions1,pos[1:,:],mask = model.crf_mask(pos[1:,:],model.pos_pad))
        loss2 = -model.crf2(predictions2,neg_scope[1:,:],mask = model.crf_mask(neg_scope[1:,:],model.neg_pad))

        predictions1 = torch.Tensor(np.array(model.crf1.decode(predictions1)).T).reshape(-1,1).to(torch.device('cuda'))
        predictions2 = torch.Tensor(np.array(model.crf2.decode(predictions2)).T).reshape(-1,1).to(torch.device('cuda'))
        # predictions1 = predictions1.view(-1, predictions1.shape[-1])
        # predictions2 = predictions2.view(-1, predictions2.shape[-1])
        pos = pos[1:,:].view(-1)
        neg_scope = neg_scope[1:,:].view(-1)       
        # loss1 = criterion(predictions1, pos) 
        # loss2 = criterion(predictions2, neg_scope)
        
        loss = loss1+loss2
                
        acc_pos = categorical_accuracy(predictions1, pos, model.pos_pad,listed =True)
        acc_neg = categorical_accuracy(predictions2, neg_scope, model.neg_pad,listed =True)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc_pos += acc_pos.item()
        epoch_acc_neg += acc_neg.item()
        
    return epoch_loss / len(iterator), epoch_acc_pos / len(iterator), epoch_acc_neg / len(iterator)

def evaluate(model, iterator, criterion):
    
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
            loss1 = -model.crf1(predictions1,pos[1:,:],mask = model.crf_mask(pos[1:,:],model.pos_pad))
            loss2 = -model.crf2(predictions2,neg_scope[1:,:],mask = model.crf_mask(pos[1:,:],model.pos_pad))

            predictions1 = torch.Tensor(np.array(model.crf1.decode(predictions1)).T).reshape(-1,1).to(torch.device('cuda'))
            predictions2 = torch.Tensor(np.array(model.crf2.decode(predictions2)).T).reshape(-1,1).to(torch.device('cuda'))
            # predictions1 = predictions1.view(-1, predictions1.shape[-1])
            # predictions2 = predictions2.view(-1, predictions2.shape[-1])
            pos = pos[1:,:].view(-1)
            neg_scope = neg_scope[1:,:].view(-1)
            
            # loss1 = criterion(predictions1, pos) 
            # loss2 = criterion(predictions2, neg_scope)

            loss = loss1+loss2

            acc_pos = categorical_accuracy(predictions1, pos, model.pos_pad,listed =True)
            acc_neg = categorical_accuracy(predictions2, neg_scope, model.neg_pad,listed =True)

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
                 input_dimA, input_dimB,text_pad,pos_pad,neg_pad):
        
        super().__init__()
        self.text_pad = text_pad
        self.pos_pad = pos_pad
        self.neg_pad = neg_pad
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = text_pad)
        self.embeddingA = nn.Embedding(input_dimA, embedding_dim, padding_idx = pos_pad)
        self.embeddingB = nn.Embedding(input_dimB, embedding_dim, padding_idx = neg_pad)
        
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers = n_layers, 
                            bidirectional = bidirectional,
                            dropout = dropout if n_layers > 1 else 0)
        self.pos_encoding = PositionalEncoding(hidden_dim * 2 if bidirectional else hidden_dim, dropout)
        self.pos_encoding_trg1 = PositionalEncoding(hidden_dim * 2 if bidirectional else hidden_dim, dropout)
        self.pos_encoding_trg2 = PositionalEncoding(hidden_dim * 2 if bidirectional else hidden_dim, dropout)
        self.T1 = torch.nn.Transformer(d_model=hidden_dim * 2 if bidirectional else hidden_dim,num_encoder_layers=3, num_decoder_layers=1)
        self.T2 = torch.nn.Transformer(d_model=hidden_dim * 2 if bidirectional else hidden_dim,num_encoder_layers=3, num_decoder_layers=1)
        
        self.fc1 = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim1)
        self.fc2 = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim2)
        self.crf1 = CRF(output_dim1)
        self.crf2 = CRF(output_dim2)
        self.dropout = nn.Dropout(dropout)
    def make_len_mask(self, inp,pad):
        return (inp == pad).transpose(0, 1)
    def crf_mask(self, inp,pad):
        return (inp != pad)

    def forward(self, text, y1, y2):
        src_pad_mask = self.make_len_mask(text,self.text_pad)
        trg_pad_mask1 = self.make_len_mask(y1,self.pos_pad)
        trg_pad_mask2 = self.make_len_mask(y2,self.neg_pad)
        embedded_X = self.dropout(self.embedding(text))
        embedded_y1 = self.dropout(self.embeddingA(y1))
        embedded_y2 = self.dropout(self.embeddingB(y2))
        shared_output, (hidden, cell) = self.lstm(embedded_X)
        src_mask = self.T1.generate_square_subsequent_mask(len(shared_output)).to(shared_output.device)
        trg_mask1 = self.T1.generate_square_subsequent_mask(len(embedded_y1)).to(embedded_y1.device)
        trg_mask2 = self.T2.generate_square_subsequent_mask(len(embedded_y2)).to(embedded_y2.device)
        shared_output = self.pos_encoding(shared_output)
        embedded_y1 = self.pos_encoding_trg1(embedded_y1)
        embedded_y2 = self.pos_encoding_trg2(embedded_y2)
        out1 = self.T1(shared_output, embedded_y1, tgt_mask=trg_mask1,src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask1, memory_key_padding_mask=src_pad_mask)#,src_mask=src_mask
        out2 = self.T2(shared_output, embedded_y2, tgt_mask=trg_mask2,src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask2, memory_key_padding_mask=src_pad_mask)#,src_mask=src_mask
        predictions_1 = self.fc1(self.dropout(out1))
        predictions_2 = self.fc2(self.dropout(out2))
        return predictions_1, predictions_2