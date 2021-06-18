from test import test_model, validate_model
import torch
import torch.nn as nn
import torch.optim as optim
from model import MyModel_2, train_model, evaluate
from torchtext import data
import numpy as np
from utils import epoch_time
import time
import random
import os
SEED = 42
from tqdm import tqdm
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

root = './training/' + 'mtl2_up/'

BOS_WORD = '<sos>'
EOS_WORD = '<eos>'
BLANK_WORD = "<blank>"
MAX_LEN = 85
TEXT = data.Field(lower = True, pad_token= BLANK_WORD,fix_length=MAX_LEN)
POS = data.Field(unk_token = None, pad_token= BLANK_WORD,init_token= BOS_WORD, eos_token= EOS_WORD,fix_length=MAX_LEN)
NEG_SCOPE = data.Field(unk_token = None, pad_token= BLANK_WORD,init_token= BOS_WORD, eos_token= EOS_WORD,fix_length=MAX_LEN)

fields = (("Sentence", TEXT), ("POS", POS), ("Neg_Scope", NEG_SCOPE))

train, val, test = data.TabularDataset.splits(path='./', train='data/train.csv',\
                                            validation='data/val.csv', test= 'data/test_cardboard.csv',\
                                            format='csv', fields=fields, skip_header=True)

TEXT.build_vocab(train, val)
POS.build_vocab(train, val)
NEG_SCOPE.build_vocab(train, val)

INPUT_DIM = len(TEXT.vocab)
INPUT_DIMA = len(POS.vocab)
INPUT_DIMB = len(NEG_SCOPE.vocab)
EMBEDDING_DIM = 256
HIDDEN_DIM = 128
OUTPUT_DIM1 = len(POS.vocab)
OUTPUT_DIM2 = len(NEG_SCOPE.vocab)
N_LAYERS = 3
BIDIRECTIONAL = True
DROPOUT = 0.25
TEXT_PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
TAG_PAD_IDX = POS.vocab.stoi[POS.pad_token]
NEG_PAD_IDX = NEG_SCOPE.vocab.stoi[NEG_SCOPE.pad_token]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MyModel_2(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM1, \
                OUTPUT_DIM2, N_LAYERS, BIDIRECTIONAL, DROPOUT, INPUT_DIMA, INPUT_DIMB,\
                    TEXT_PAD_IDX, TAG_PAD_IDX, NEG_PAD_IDX)
model = model.to(device)
# model.load_state_dict(torch.load('./training/' + 'mtl2/' + 'ep-100.pt'))
optimizer = optim.Adam(model.parameters(),lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX)


# BATCH_SIZE = 64

# train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
#     (train, val, test), 
#     batch_size = BATCH_SIZE,
#     device = device,sort=False)

# # bat = next(iter(train_iterator))
# # from torchviz import make_dot
# # out1, out2 = model(bat.Sentence,bat.POS,bat.Neg_Scope)
# # make_dot(out1) 
# N_EPOCHS = 200
# best_valid_loss = float('inf')

# for epoch in tqdm(range(N_EPOCHS)):

#     start_time = time.time()
    
#     train_loss, train_acc_pos, train_acc_neg = train_model(model, train_iterator, optimizer, criterion)
#     valid_loss, valid_acc_pos, valid_acc_neg = evaluate(model, valid_iterator, criterion)
    
#     end_time = time.time()

#     epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
#     # if valid_loss < best_valid_loss:
#     #     best_valid_loss = valid_loss
#     if(epoch%25==0):
#         torch.save(model.state_dict(), root + f'ep-{epoch}.pt')
#     scheduler.step(valid_loss)
#     print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s',flush=True)
#     print(f'\tTrain Loss: {train_loss:.3f} | Train Acc POS: {train_acc_pos*100:.2f}% | Train Acc NEG: {train_acc_neg*100:.2f}%' ,flush=True)
#     print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc POS: {valid_acc_pos*100:.2f}% | Val. Acc NEG: {valid_acc_neg*100:.2f}%',flush=True)
# torch.save(model.state_dict(), root + f'ep-{epoch}.pt')
# TESTING

BATCH_SIZE = len(test)

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train, val, test), 
    batch_size = BATCH_SIZE,
    device = device,sort=False)

model.load_state_dict(torch.load(root + 'ep-199.pt'))
model = model.to(device)
TAG_NEG_IDX  = NEG_SCOPE.vocab.stoi['1']
test_loss, test_acc_pos, test_acc_neg = test_model(model, test_iterator, criterion, TAG_PAD_IDX, TAG_NEG_IDX)
print(f'\t Test. Loss: {test_loss:.3f} |  Test. Acc POS: {test_acc_pos*100:.2f}% | Test. Acc NEG: {test_acc_neg*100:.2f}%',flush=True)

BATCH_SIZE = 1

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train, val, test), 
    batch_size = BATCH_SIZE,
    device = device,sort=False)

test_acc_pos, test_acc_neg = validate_model(model, test_iterator, criterion, TAG_PAD_IDX, TAG_NEG_IDX)
print(f'\t Test. Acc POS: {test_acc_pos*100:.2f}% | Test. Acc NEG: {test_acc_neg*100:.2f}%',flush=True)