from sklearn.metrics import f1_score
import torch

# def create_vocab(training_sentences, training_POS):
#     words, tags = set([]), set([]) 
#     for s in training_sentences:
#         for w in s:
#             words.add(w.lower())
#     for ts in training_POS:
#         for t in ts:
#             tags.add(t)
#     word2index = {w: i + 2 for i, w in enumerate(list(words))}
#     word2index['-PAD-'] = 0  # The special value used for padding
#     word2index['-OOV-'] = 1  # The special value used for OOVs
#     tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
#     tag2index['-PAD-'] = 0  # The special value used to padding
#     return words, tags, word2index, tag2index

# def convert2index(sentences,word2index,POS,tag2index):
#     sentences_X = []
#     tags_y = []
#     for s in sentences:
#         s_int = []
#         for w in s:
#             try:
#                 s_int.append(word2index[w.lower()])
#             except KeyError:
#                 s_int.append(word2index['-OOV-'])
    
#         sentences_X.append(s_int)
#     for s in POS:
#         tags_y.append([tag2index[t] for t in s])

#     return sentences_X, tags_y

def f1_scope(y_true, y_pred, level = 'scope'): #This is for gold cue annotation scope, thus the precision is always 1.
    if level == 'token':
        print(f1_score([i for i in j for j in y_true], [i for i in j for j in y_pred]))
    elif level == 'scope':
        tp = 0
        fn = 0
        fp = 0
        for y_t, y_p in zip(y_true, y_pred):
            if y_t == y_p:
                tp+=1
            else:
                fn+=1
        prec = 1
        rec = tp/(tp+fn)
        print(f"Precision: {prec}")
        print(f"Recall: {rec}")
        print(f"F1 Score: {2*prec*rec/(prec+rec)}")

def categorical_accuracy(preds, y, tag_pad_idx,listed=False):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    if(not listed):
        max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    else:
        max_preds = preds
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]]).to(torch.device('cuda'))

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def f1(preds, y, tag_pad_idx, cls,listed=False):

    if(not listed):
        max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    else:
        max_preds = preds
    non_pad_elements = (y != tag_pad_idx).nonzero()
    # correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    y_hat = max_preds[non_pad_elements].squeeze(1)
    y_real = y[non_pad_elements]
    counter =dict(zip(* torch.unique(y_hat,return_counts=True)))
    for k,v in list(counter.items()):
        counter[k.item()]=v.item()
    # counter = counter.to(torch.device('cuda'))
    try:
        if(counter[cls] != 0):
            P = len(y_real[(y_real == y_hat)  & (y_real == cls) & (y_hat == cls)])/counter[cls]
    except:
        P = 0.001
        print(P)
        pass
    counter = dict(zip(*torch.unique(y_real,return_counts=True)))
    for k,v in list(counter.items()):
        counter[k.item()]=v.item()
    # counter = counter.to(torch.device('cuda'))
    try:
        if(counter[cls] != 0):
            R = len(y_real[(y_real == y_hat)  & (y_real == cls) & (y_hat == cls)])/counter[cls]
    except:
        R = 0.001
        print(R)
        pass
    
    return 2*P*R/(P+R)
