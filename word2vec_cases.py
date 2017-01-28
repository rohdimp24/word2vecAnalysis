import multiprocessing
import os,sys
import pickle
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.porter import PorterStemmer
import gensim, logging
import re


tknzr = WordPunctTokenizer()
nltk.download('stopwords')
stoplist = stopwords.words('english')
stemmer = PorterStemmer()


#
# class MySentences(object):
#     def __init__(self, dirname):
#         self.dirname = dirname
#
#     def __iter__(self):
#         for fname in os.listdir(self.dirname):
#             for line in open(os.path.join(self.dirname, fname)):
#                 yield line.split()

#sentences = MySentences('/predix/PycharmProjects/nltkRohit/cases') # a memory-friendly iterator





def review_to_wordlist( review_text, remove_stopwords=True ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    #review_text = BeautifulSoup(review).get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
        words = [w for w in words if len(w)>1]
    #
    # 5. Return a list of words
    return(words)

def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences





tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')





def PreprocessDoc2Vec(text,stopwordList,removeStopwords=True):
    #text="After the machine came back online on 1 April the vibrations on the outboard bearing increased by ~ 0.3 mil in both directions."
    print(text)
    text = re.sub("[^a-zA-Z]"," ", text)
    text=text.lower()
    words = tknzr.tokenize(text)
    words = [w for w in words if len(w)>1]
    if removeStopwords:
        words_clean = [i.lower() for i in words if i not in stopwordList]
    else:
        words_clean=words
    return words_clean



''''
sentences = []  # Initialize an empty list of sentences
dir = '/predix/PycharmProjects/nltkRohit/all'
for filename in [f for f in os.listdir(dir) if str(f)[0]!='.']:
    print(filename)
    f = open(dir+'/'+filename,'r')
    sentences.append(PreprocessDoc2Vec(f.read(),stopwordList))

'''
#=================

# import modules & set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#sentences = [['first', 'sentence'], ['second', 'sentence']]
# train word2vec on the two sentences

#reading the sentneces using the line by line method versus reading from the individual files has different computation times
sentenceIterator = gensim.models.word2vec.LineSentence("/Users/305015992/pythonProjects/word2vecAnalysis/all_jim_case_large.txt")


#stopwords
stopwordsFile = open('/Users/305015992/pythonProjects/word2vecAnalysis/stopwords.txt', 'r')
stopwords=stopwordsFile.read()
#stopwords=stopwords.lower()
stopwordList=stopwords.split(",")
print(stopwordList)

##this is used when we read the single file
sentences = []
#count=0;
for sentence in sentenceIterator:
    #tt=PreprocessDoc2Vec(" ".join(sentence),stopwordList)
    #print(tt)
    sentences.append(PreprocessDoc2Vec(" ".join(sentence),stopwordList,False))
    #sentences.append(tt)
    # count+=1
    #     #sentences.append(PreprocessDoc2Vec(sentence,stoplist))
    # if(count>120):
    #     break;

print(len(sentences))
# print(sentences[15507])

###if you want bigrams
# bigram_transformer = gensim.models.Phrases(sentences)
# model = gensim.models.Word2Vec(bigram_transformer[sentences], min_count=10, size=100,sg=1,hs=1)



#skipgram with hierarichal softmax and the dense vector of dimention 50 inital the word vectoir is |Vocab| and we convert it
# to |50|
#Word2Vec expects single sentences, each one as a list of words. In other words, the input format is a list of lists. ..https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-2-word-vectors
#also it suggests that we should not use the stopwords removal. I think we can first run the model and then while displaying remove the unncessary stuff

model = gensim.models.Word2Vec(sentences, min_count=5, size=50,sg=1,hs=1)

#CBOW..this is giving all the similarity scores as 0.99
# model1 = gensim.models.Word2Vec(sentences, min_count=5, size=50)



#train the model 20 times
numEpochs=20
for epoch in range(numEpochs):
    try:
        print('epoch %d'%(epoch))
        model.train(sentences)
    except(KeyboardInterrupt,SystemExit):
        break



#access all the words in the vocabulary
vocab = list(model.vocab.keys())
len(vocab)

print(vocab)
print(stopwordList)

#need to remove the words which are stop words from the vocabulary
vocab=[i for i in vocab if i not in stopwordList]
print(len(vocab))


#vocab.remove('abs')

#this gives the dense vector
model['turbine']


'''
1. we need to remove the words that are stopwords from the vocabulary
2. for each of the list item we will run a check to remove the listing of the stopwords
The only proble i see in this approach where we apply the stopwords after the training is that the probability is going down..we need to verify with the
previous approaches
'''
def getRefinedOutput(model,word):
    output=[]
    tt=model.most_similar(word,topn=40)
    #now applyt the stopwrods
    for t in tt:
        if (t[0] not in stopwordList):
            output.append(t)

    return(output)


print(getRefinedOutput(model,"hp"))

#model1.most_similar('wind',topn=20)

def findSimilar(model,word):
    return(model.most_similar(word,topn=20))



###fidn the simialrity for all the vocab words and write it to a file
#fp=open('output_case1','w')
result={}
for word in vocab:
   # fp.write(word+":")
   # print(word)
   # print(findSimilar(model,word))
   # fp.write(str(findSimilar(model,word))+os.linesep)
   #  result[word]=(findSimilar(model,word))
    result[word]=(getRefinedOutput(model,word))

#fp.close()

import json
fp=open('output_case_break_with_tolower_withoutStopwords.json','w')
fp.write(json.dumps(result))
fp.close()
##genrate the tsne plot

def generateTSNEPlot(vocab,model):

    word_vectors=[]
    #collect all the word vectors
    for word in vocab:
        print(word)
        word_vectors.append(model[word])

    import numpy as np
    from sklearn.manifold import TSNE


    #http://hen-drik.de/pub/Heuer%20-%20word2vec%20-%20From%20theory%20to%20practice.pdf
    #
    # X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    # model = TSNE(n_components=2, random_state=0)
    # np.set_printoptions(suppress=True)
    # model.fit_transform(X)
    import matplotlib.pyplot as plt
    vectors=np.asfarray(word_vectors,dtype='float')
    tnse=TSNE(n_components=2, random_state=0)
    tt=tnse.fit_transform(vectors)

    fig, ax = plt.subplots()

    for i, txt in enumerate(vocab):
        # print(i,txt)
        ax.annotate(txt, (tt[i,0],tt[i,1]))

    plt.scatter(tt[:, 0], tt[:, 1])


    #https://raw.githubusercontent.com/prateekpg2455/U.S-Presidential-Speeches/master/w2v_speech.py he talks that we need to take the average score for sentence
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



#use this function to plot instead of tsne
def plotData(tfSparseMatrix):
    X=tfSparseMatrix.todense()
    pca = PCA(n_components=2).fit(X)
    data2D = pca.transform(X)
    plt.scatter(data2D[:,0], data2D[:,1])
    plt.show()


models=[]
for i in range(100):
    models.append(model[vocab[i]])


import scipy as sp
import numpy as np
sparse_m = sp.sparse.bsr_matrix(np.array(models))

plotData(sparse_m)

# generateTSNEPlot(vocab[1:5],model)


##tests
