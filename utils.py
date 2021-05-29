from sklearn.metrics import f1_score
def create_vocab(training_sentences, training_POS):
    words, tags = set([]), set([]) 
    for s in training_sentences:
        for w in s:
            words.add(w.lower())
    for ts in training_POS:
        for t in ts:
            tags.add(t)
    word2index = {w: i + 2 for i, w in enumerate(list(words))}
    word2index['-PAD-'] = 0  # The special value used for padding
    word2index['-OOV-'] = 1  # The special value used for OOVs
    tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
    tag2index['-PAD-'] = 0  # The special value used to padding
    return words, tags, word2index, tag2index

def convert2index(sentences,word2index,POS,tag2index):
    sentences_X = []
    tags_y = []
    for s in sentences:
        s_int = []
        for w in s:
            try:
                s_int.append(word2index[w.lower()])
            except KeyError:
                s_int.append(word2index['-OOV-'])
    
        sentences_X.append(s_int)
    for s in POS:
        tags_y.append([tag2index[t] for t in s])

    return sentences_X, tags_y

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