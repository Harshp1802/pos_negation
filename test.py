from utils import categorical_accuracy, f1
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

def test_model(model, iterator, criterion, tag_pad_idx, tag_neg_idx):
    model.eval()
    assert len(iterator)==1
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

            acc_pos = categorical_accuracy(predictions1, pos, model.pos_pad,listed =True).item()
            acc_neg = f1(predictions2, neg_scope, model.neg_pad,tag_neg_idx,listed =True)
            # acc_pos = categorical_accuracy(predictions1, pos, tag_pad_idx).item()
            # acc_neg = f1(predictions2, neg_scope, tag_pad_idx, tag_neg_idx)

    return loss / len(iterator), acc_pos / len(iterator), acc_neg / len(iterator)

def validate_model(model, iterator, criterion, tag_pad_idx, tag_neg_idx):
    model.eval()
    with torch.no_grad():
        preds1 = []
        preds2 = []
        y1s = []
        y2s = []
        for batch in tqdm(iterator):
            text = batch.Sentence
            pos = batch.POS
            neg_scope = batch.Neg_Scope
            src_pad_mask = model.make_len_mask(text)
            
            embedded_X = model.dropout(model.embedding(text))
            shared_output, (hidden, cell) = model.lstm(embedded_X)
            shared_output = model.pos_encoding(shared_output)
            src_mask = model.T1.generate_square_subsequent_mask(len(shared_output)).to(shared_output.device)
            memory1 = model.T1.encoder(shared_output,src_key_padding_mask=src_pad_mask,mask=src_mask)
            memory2 = model.T2.encoder(shared_output,src_key_padding_mask=src_pad_mask,mask=src_mask)

            y1_out_indexes = [1,]
            y2_out_indexes = [1,]

            for i in range(pos.shape[0]-1):
                trg_tensor1 = torch.LongTensor(y1_out_indexes).unsqueeze(1).to(memory1.device)
                trg_tensor2 = torch.LongTensor(y2_out_indexes).unsqueeze(1).to(memory2.device)
                trg_pad_mask1 = model.make_len_mask(trg_tensor1)
                trg_pad_mask2 = model.make_len_mask(trg_tensor2)
                trg_mask1 = model.T1.generate_square_subsequent_mask(len(trg_tensor1)).to(trg_tensor1.device)
                trg_mask2 = model.T2.generate_square_subsequent_mask(len(trg_tensor2)).to(trg_tensor2.device)
                output1 = model.fc1(model.dropout(model.T1.decoder(model.dropout(model.embeddingA(trg_tensor1)), memory1, tgt_key_padding_mask=trg_pad_mask1,memory_key_padding_mask=src_pad_mask,tgt_mask=trg_mask1)))
                output2 = model.fc2(model.dropout(model.T2.decoder(model.dropout(model.embeddingB(trg_tensor2)), memory2, tgt_key_padding_mask=trg_pad_mask2,memory_key_padding_mask=src_pad_mask,tgt_mask=trg_mask2)))
                out_token1 = output1.argmax(2)[-1].item()
                out_token2 = output2.argmax(2)[-1].item()
                y1_out_indexes.append(out_token1)
                y2_out_indexes.append(out_token2)
            preds1.append(y1_out_indexes)
            preds2.append(y2_out_indexes)
            y1s.append(pos)
            y2s.append(neg_scope)

        preds1 = torch.Tensor(np.array(preds1)[:,1:].T).reshape(-1,1).to(torch.device('cuda'))
        y1s = torch.cat(y1s, dim=1)[1:,:].view(-1).to(torch.device('cuda'))
        preds2 = torch.Tensor(np.array(preds2)[:,1:].T).reshape(-1,1).to(torch.device('cuda'))
        y2s = torch.cat(y2s, dim=1)[1:,:].view(-1).to(torch.device('cuda'))
        acc_pos = categorical_accuracy(preds1, y1s, tag_pad_idx,listed =True).item()
        acc_neg = f1(preds2, y2s, tag_pad_idx, tag_neg_idx,listed =True)

    return acc_pos, acc_neg