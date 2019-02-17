
# coding: utf-8

# In[117]:


from collections import Counter
import itertools
import string as string

import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import linalg 
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


# In[128]:


df = []
with open('./input/bible.txt') as f:
    for line in f:
        df.append(line[0:(len(line)-1)])
df[0:5]


# In[136]:


headlines = df


# In[130]:


#df = pd.read_csv('./input/abcnews-date-text.csv')
#df = pd.read_csv('./input/Amazon_Unlocked_Mobile.csv')
#df.head()


# In[67]:


#df.headline_text
#df.Reviews


# In[137]:


string.punctuation


# In[138]:


#headlines = df['headline_text'].tolist()


# In[139]:


headlines = [str(headline) for headline in headlines]


# In[140]:


table = str.maketrans('', '', string.punctuation)
headlines = [headline.translate(table) for headline in headlines]


# In[141]:


headlines = [headline.lower() for headline in headlines]


# In[142]:


headlines[0:5]


# In[143]:


#headlines = df['headline_text'].tolist()
# remove stopwords
stopwords_set = set(stopwords.words('english'))
headlines = [
    [tok for tok in headline.split() if tok not in stopwords_set] for headline in headlines
]
# remove single word headlines
headlines = [hl for hl in headlines if len(hl) > 1]
# show results
headlines[0:20]


# In[144]:


tok2indx = dict()
unigram_counts = Counter()
for ii, headline in enumerate(headlines):
    if ii % 200000 == 0:
        print(f'finished {ii/len(headlines):.2%} of headlines')
    for token in headline:
        unigram_counts[token] += 1
        if token not in tok2indx:
            tok2indx[token] = len(tok2indx)
indx2tok = {indx:tok for tok,indx in tok2indx.items()}
print('done')
print('vocabulary size: {}'.format(len(unigram_counts)))
print('most common: {}'.format(unigram_counts.most_common(10)))
vocab = len(unigram_counts)


# In[145]:


# note add dynammic window hyperparameter
back_window = 2
front_window = 2
skipgram_counts = Counter()
for iheadline, headline in enumerate(headlines):
    for ifw, fw in enumerate(headline):
        icw_min = max(0, ifw - back_window)
        icw_max = min(len(headline) - 1, ifw + front_window)
        icws = [ii for ii in range(icw_min, icw_max + 1) if ii != ifw]
        for icw in icws:
            skipgram = (headline[ifw], headline[icw])
            skipgram_counts[skipgram] += 1    
    if iheadline % 200000 == 0:
        print(f'finished {iheadline/len(headlines):.2%} of headlines')
        
print('done')
print('number of skipgrams: {}'.format(len(skipgram_counts)))
print('most common: {}'.format(skipgram_counts.most_common(10)))


# In[146]:


row_indxs = []
col_indxs = []
dat_values = []
ii = 0
for (tok1, tok2), sg_count in skipgram_counts.items():
    ii += 1
    if ii % 1000000 == 0:
        print(f'finished {ii/len(skipgram_counts):.2%} of skipgrams')
    tok1_indx = tok2indx[tok1]
    tok2_indx = tok2indx[tok2]
        
    row_indxs.append(tok1_indx)
    col_indxs.append(tok2_indx)
    dat_values.append(sg_count)
    
wwcnt_mat = sparse.csr_matrix((dat_values, (row_indxs, col_indxs)))
print('done')


# In[147]:


def ww_sim(word, mat, topn=10):
    """Calculate topn most similar words to word"""
    indx = tok2indx[word]
    if isinstance(mat, sparse.csr_matrix):
        v1 = mat.getrow(indx)
    else:
        v1 = mat[indx:indx+1, :]
    sims = cosine_similarity(mat, v1).flatten()
    sindxs = np.argsort(-sims)
    sim_word_scores = [(indx2tok[sindx], sims[sindx]) for sindx in sindxs[0:topn]]
    return sim_word_scores


# In[148]:


ww_sim('strike', wwcnt_mat)
# need to also take out ; when we remove stopwords


# In[149]:


wwcnt_norm_mat = normalize(wwcnt_mat, norm='l2', axis=1)


# In[150]:


ww_sim('strike', wwcnt_norm_mat)


# In[151]:


num_skipgrams = wwcnt_mat.sum()
assert(sum(skipgram_counts.values())==num_skipgrams)

# for creating sparce matrices
row_indxs = []
col_indxs = []

pmi_dat_values = []
ppmi_dat_values = []
spmi_dat_values = []
sppmi_dat_values = []

# smoothing
alpha = 0.75
nca_denom = np.sum(np.array(wwcnt_mat.sum(axis=0)).flatten()**alpha)
sum_over_words = np.array(wwcnt_mat.sum(axis=0)).flatten()
sum_over_words_alpha = sum_over_words**alpha
sum_over_contexts = np.array(wwcnt_mat.sum(axis=1)).flatten()

ii = 0
for (tok1, tok2), sg_count in skipgram_counts.items():
    ii += 1
    if ii % 1000000 == 0:
        print(f'finished {ii/len(skipgram_counts):.2%} of skipgrams')
    tok1_indx = tok2indx[tok1]
    tok2_indx = tok2indx[tok2]
    
    nwc = sg_count
    Pwc = nwc / num_skipgrams
    nw = sum_over_contexts[tok1_indx]
    Pw = nw / num_skipgrams
    nc = sum_over_words[tok2_indx]
    Pc = nc / num_skipgrams
    
    nca = sum_over_words_alpha[tok2_indx]
    Pca = nca / nca_denom
    
    pmi = np.log2(Pwc/(Pw*Pc))
    ppmi = max(pmi, 0)
    
    spmi = np.log2(Pwc/(Pw*Pca))
    sppmi = max(spmi, 0)
    
    row_indxs.append(tok1_indx)
    col_indxs.append(tok2_indx)
    pmi_dat_values.append(pmi)
    ppmi_dat_values.append(ppmi)
    spmi_dat_values.append(spmi)
    sppmi_dat_values.append(sppmi)
        
pmi_mat = sparse.csr_matrix((pmi_dat_values, (row_indxs, col_indxs)))
ppmi_mat = sparse.csr_matrix((ppmi_dat_values, (row_indxs, col_indxs)))
spmi_mat = sparse.csr_matrix((spmi_dat_values, (row_indxs, col_indxs)))
sppmi_mat = sparse.csr_matrix((sppmi_dat_values, (row_indxs, col_indxs)))

print('done')


# In[152]:


ww_sim('strike', pmi_mat)


# In[153]:


ww_sim('phone', ppmi_mat)


# In[ ]:


ww_sim('phone', spmi_mat)


# In[ ]:


ww_sim('phone', sppmi_mat)


# In[154]:


pmi_use = ppmi_mat
embedding_size = 50
uu, ss, vv = linalg.svds(pmi_use, embedding_size)


# In[155]:


print('vocab size: {}'.format(len(unigram_counts)))
print('embedding size: {}'.format(embedding_size))
print('uu.shape: {}'.format(uu.shape))
print('ss.shape: {}'.format(ss.shape))
print('vv.shape: {}'.format(vv.shape))


# In[156]:


unorm = uu / np.sqrt(np.sum(uu*uu, axis=1, keepdims=True))
vnorm = vv / np.sqrt(np.sum(vv*vv, axis=0, keepdims=True))
#word_vecs = unorm
#word_vecs = vnorm.T
word_vecs = uu + vv.T
word_vecs_norm = word_vecs / np.sqrt(np.sum(word_vecs*word_vecs, axis=1, keepdims=True))


# In[157]:


def word_sim_report(word, sim_mat):
    sim_word_scores = ww_sim(word, word_vecs)
    for sim_word, sim_score in sim_word_scores:
        print(sim_word, sim_score)
        word_headlines = [hl for hl in headlines if sim_word in hl and word in hl][0:5]
        for headline in word_headlines:
            print(f'    {headline}')


# In[159]:


word = 'smite'
word_sim_report(word, word_vecs)


# In[160]:


word = 'new'
word_sim_report(word, word_vecs)


# In[163]:


word = 'jesus'
word_sim_report(word, word_vecs)


# In[165]:


word = 'ass'
word_sim_report(word, word_vecs)


# In[166]:


word = 'flesh'
word_sim_report(word, word_vecs)


# In[168]:


word = 'love'
word_sim_report(word, word_vecs)


# In[169]:


def wvec_sim(vec, mat, topn=10):
    """Calculate topn most similar words to vec"""
    v1 = vec
    sims = cosine_similarity(mat, v1).flatten()
    sindxs = np.argsort(-sims)
    sim_word_scores = [(indx2tok[sindx], sims[sindx]) for sindx in sindxs[0:topn]]
    return sim_word_scores


# In[102]:


#print(vv[0].reshape(-1,1))


# In[103]:


#wvec_sim(sparse.csr_matrix((vv[0], (np.zeros(107509), np.arange(107509))), shape=(1, 107509)), ppmi_mat)


# In[170]:


for i in range(0,50):
    print(i)
    print(ss[i])
    print(wvec_sim(sparse.csr_matrix((vv[i], (np.zeros(vocab), np.arange(vocab))), shape=(1, vocab)), ppmi_mat)[0:5])


# In[171]:


# 46 charged jailed
# 44 interview
# 43 govt
# 41 scientists
#vv[47] + vv[39]
print(wvec_sim(sparse.csr_matrix((vv[41]+vv[46], (np.zeros(vocab), np.arange(vocab))), shape=(1, vocab)), ppmi_mat)[0:10])


# In[172]:


print(wvec_sim(sparse.csr_matrix((vv[0], (np.zeros(vocab), np.arange(vocab))), shape=(1, vocab)), ppmi_mat)[0:10])
print(wvec_sim(sparse.csr_matrix((vv[1], (np.zeros(vocab), np.arange(vocab))), shape=(1, vocab)), ppmi_mat)[0:10])
print(wvec_sim(sparse.csr_matrix((vv[2], (np.zeros(vocab), np.arange(vocab))), shape=(1, vocab)), ppmi_mat)[0:10])
print(wvec_sim(sparse.csr_matrix((vv[3], (np.zeros(vocab), np.arange(vocab))), shape=(1, vocab)), ppmi_mat)[0:10])


# In[173]:


def ww_sim_print(word, mat, topn=10):
    """Calculate topn most similar words to word"""
    indx = tok2indx[word]
    print(indx)
    if isinstance(mat, sparse.csr_matrix):
        v1 = mat.getrow(indx)
    else:
        v1 = mat[indx:indx+1, :]
    print(type(v1))
    print(v1)
    print(v1.toarray().shape)
    sims = cosine_similarity(mat, v1).flatten()
    sindxs = np.argsort(-sims)
    sim_word_scores = [(indx2tok[sindx], sims[sindx]) for sindx in sindxs[0:topn]]
    return 0 #sim_word_scores


# In[174]:


ss


# In[175]:


def ww_sim(word, mat, topn=10):
    """Calculate topn most similar words to word"""
    indx = tok2indx[word]
    if isinstance(mat, sparse.csr_matrix):
        v1 = mat.getrow(indx)
    else:
        v1 = mat[indx:indx+1, :]
    sims = cosine_similarity(mat, v1).flatten()
    sindxs = np.argsort(-sims)
    sim_word_scores = [(indx2tok[sindx], sims[sindx]) for sindx in sindxs[0:topn]]
    return sim_word_scores


# In[176]:


word = 'war'
indx = tok2indx[word]
v1 = ppmi_mat.getrow(indx)
v1


# In[199]:


# word addition
def word_addition(word1, word2, mat, add = 1):
    indx1 = tok2indx[word1]
    v1 = ppmi_mat.getrow(indx1)
    indx2 = tok2indx[word2]
    v2 = ppmi_mat.getrow(indx2)
    #indx3 = tok2indx[word3]
    #v3 = ppmi_mat.getrow(indx3)
    return(wvec_sim(v1 + add * v2, mat))


# In[206]:


print(word_addition('speaking', 'tongues', ppmi_mat, 1))

