import random

class Cues:
    def __init__(self, data):
        self.sentences = data[0]
        self.cues = data[1]
        self.num_sentences = len(data[0])
class Scopes:
    def __init__(self, data):
        self.sentences = data[0]
        self.cues = data[1]
        self.scopes = data[2]
        self.pos = data[3]
        self.num_sentences = len(data[0])

def starsem(f_path, cue_sents_only=False, frac_no_cue_sents = 1.0):
    raw_data = open(f_path)
    sentence = []
    labels = []
    label = []
    scope_sents = []
    scope_pos =[]
    data_scope = []
    scope = []
    scope_cues = []
    data = []
    cue_only_data = []
    POS = []
    
    for line in raw_data:
        label = []
        sentence = []
        POS = []
        tokens = line.strip().split()
        if len(tokens)==8: #This line has no cues
                sentence.append(tokens[3])
                POS.append(tokens[5])
                label.append(3) #Not a cue
                for line in raw_data:
                    tokens = line.strip().split()
                    if len(tokens)==0:
                        break
                    else:
                        sentence.append(tokens[3])
                        POS.append(tokens[5])
                        label.append(3)
                cue_only_data.append([sentence, label, POS])
                
            
        else: #The line has 1 or more cues
            num_cues = (len(tokens)-7)//3
            #cue_count+=num_cues
            scope = [[] for i in range(num_cues)]
            label = [[],[]] #First list is the real labels, second list is to modify if it is a multi-word cue.
            label[0].append(3) #Generally not a cue, if it is will be set ahead.
            label[1].append(-1) #Since not a cue, for now.
            for i in range(num_cues):
                if tokens[7+3*i] != '_': #Cue field is active
                    if tokens[8+3*i] != '_': #Check for affix
                        label[0][-1] = 0 #Affix
                        affix_list.append(tokens[7+3*i])
                        label[1][-1] = i #Cue number
                        #sentence.append(tokens[7+3*i])
                        #new_word = '##'+tokens[8+3*i]
                    else:
                        label[0][-1] = 1 #Maybe a normal or multiword cue. The next few words will determine which.
                        label[1][-1] = i #Which cue field, for multiword cue altering.
                        
                if tokens[8+3*i] != '_':
                    scope[i].append(1)
                else:
                    scope[i].append(0)
            sentence.append(tokens[3])
            POS.append(tokens[5])
            for line in raw_data:
                tokens = line.strip().split()
                if len(tokens)==0:
                    break
                else:
                    sentence.append(tokens[3])
                    POS.append(tokens[5])
                    label[0].append(3) #Generally not a cue, if it is will be set ahead.
                    label[1].append(-1) #Since not a cue, for now.   
                    for i in range(num_cues):
                        if tokens[7+3*i] != '_': #Cue field is active
                            if tokens[8+3*i] != '_': #Check for affix
                                label[0][-1] = 0 #Affix
                                label[1][-1] = i #Cue number
                            else:
                                label[0][-1] = 1 #Maybe a normal or multiword cue. The next few words will determine which.
                                label[1][-1] = i #Which cue field, for multiword cue altering.
                        if tokens[8+3*i] != '_':
                            scope[i].append(1)
                        else:
                            scope[i].append(0)
            for i in range(num_cues):
                indices = [index for index,j in enumerate(label[1]) if i==j]
                count = len(indices)
                if count>1:
                    for j in indices:
                        label[0][j] = 2
            for i in range(num_cues):
                sc = []
                for a,b in zip(label[0],label[1]):
                    if i==b:
                        sc.append(a)
                    else:
                        sc.append(3)
                scope_cues.append(sc)
                scope_sents.append(sentence)
                scope_pos.append(POS)
                data_scope.append(scope[i])
            labels.append(label[0])
            data.append(sentence)
    cue_only_samples = random.sample(cue_only_data, k=int(frac_no_cue_sents*len(cue_only_data)))
    cue_only_sents = [i[0] for i in cue_only_samples]
    cue_only_cues = [i[1] for i in cue_only_samples]
    starsem_cues = (data+cue_only_sents,labels+cue_only_cues)
    starsem_scopes = (scope_sents, scope_cues, data_scope, scope_pos)
    return [starsem_cues, starsem_scopes]

if __name__ == "__main__":
    import pickle
    # Training Data:
    ret_val = starsem(r'..\starsem-st-2012-data\cd-sco\corpus\training\SEM-2012-SharedTask-CD-SCO-training-09032012.txt', frac_no_cue_sents=1.0)
    # cue_data = Cues(ret_val[0])
    scope_data = Scopes(ret_val[1])
    train_out = [scope_data.sentences, scope_data.pos, scope_data.scopes]

    pickle.dump(train_out, open(r".\data\training_starsem_pos_scope.p", "wb" ) )

    # Test Data-1:
    ret_val = starsem(r'..\starsem-st-2012-data\cd-sco\corpus\test-gold\SEM-2012-SharedTask-CD-SCO-test-cardboard-GOLD.txt', frac_no_cue_sents=1.0)
    # cue_data = Cues(ret_val[0])
    scope_data = Scopes(ret_val[1])
    test_out = [scope_data.sentences, scope_data.pos, scope_data.scopes]

    pickle.dump(test_out, open(r".\data\testing_starsem_pos_scope_1.p", "wb" ) )

    # Test Data-2:
    ret_val = starsem(r'..\starsem-st-2012-data\cd-sco\corpus\test-gold\SEM-2012-SharedTask-CD-SCO-test-circle-GOLD.txt', frac_no_cue_sents=1.0)
    # cue_data = Cues(ret_val[0])
    scope_data = Scopes(ret_val[1])
    test_out = [scope_data.sentences, scope_data.pos, scope_data.scopes]

    pickle.dump(test_out, open(r".\data\testing_starsem_pos_scope_2.p", "wb" ) )