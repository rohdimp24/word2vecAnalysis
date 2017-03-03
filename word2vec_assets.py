import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.porter import PorterStemmer
import gensim, logging
import re
import pickle
import scipy as sp
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


tknzr = WordPunctTokenizer()
# import modules & set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
stopWordsFilteringAfterModelCreation=True


'''
Preprocessing step..
Optionally you can apply the stopword processing in which case the performPOstProcessing Flag will be false
'''
def PreprocessDoc2Vec(text,stopwordList,stopWordsFilteringAfterModelCreation=True):
    print(text)
    text = re.sub("[^a-zA-Z]"," ", text)
    text=text.lower()
    words = tknzr.tokenize(text)
    words = [w for w in words if len(w)>1]
    #the stopwords will be applied later
    if stopWordsFilteringAfterModelCreation:
        words_clean = words
    #the stop words are applied before the algorithm runs
    else:
        words_clean=[i.lower() for i in words if i not in stopwordList]
    return words_clean




'''Read the stop words'''
def getStopWords():
    stopwordsFile = open('/Users/305015992/pythonProjects/word2vecAnalysis/stopwords.txt', 'r')
    stopwords=stopwordsFile.read()
    #stopwords=stopwords.lower()
    stopwordList=stopwords.split(",")
    print(stopwordList)
    return(stopwordList)

'''
1. we need to remove the words that are stopwords from the vocabulary
2. for each of the list item we will run a check to remove the listing of the stopwords
The only proble i see in this approach where we apply the stopwords after the training is that the probability is going down..we need to verify with the
previous approaches
'''
def findSimilar(model,word,stopWordsFilteringAfterModelCreation=True):
    #if you had not applied the stop words filtering before creating the model
    if(stopWordsFilteringAfterModelCreation):
        output=[]
        lstSimilarWords=model.most_similar(word,topn=40)
        #now filter out the words which are the stopwrods
        for lstItem in lstSimilarWords:
            if (lstItem[0] not in stopwordList):
                output.append(lstItem)
        return(output)
    else:
        return (model.most_similar(word, topn=20))



'''Read the cases file using the line iterator'''
sentenceIterator = gensim.models.word2vec.LineSentence("/Users/305015992/pythonProjects/word2vecAnalysis/assetNames.txt")


'''for the word2vec we want to have a list[list]] so basically each sentence is broken into tokens and we have a list of such tokens'''
sentences = []
stopwordList=getStopWords()
for sentence in sentenceIterator:
    sentences.append(PreprocessDoc2Vec(" ".join(sentence),stopwordList,stopWordsFilteringAfterModelCreation))

print(len(sentences))

'''Create the word2vec model'''
#skipgram with hierarichal softmax and the dense vector of dimention 50 inital the word vectoir is |Vocab| and we convert it
# to |50|
#Word2Vec expects single sentences, each one as a list of words. In other words, the input format is a list of lists. ..https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-2-word-vectors
#also it suggests that we should not use the stopwords removal. I think we can first run the model and then while displaying remove the unncessary stuff

model = gensim.models.Word2Vec(sentences, min_count=5, size=50,sg=1,hs=1)



#train the model 20 times
numEpochs=20
for epoch in range(numEpochs):
    try:
        print('epoch %d'%(epoch))
        model.train(sentences)
    except(KeyboardInterrupt,SystemExit):
        break


'''build the vocabulary'''
vocab = list(model.vocab.keys())
len(vocab)


'''Post processing step'''

#remove the words that are presnt in the stopwords
vocab=[i for i in vocab if i not in stopwordList]
print(len(vocab))

print(vocab)

#this gives the dense vector
model['pump']



# print(findSimilar(model,"hp"))
#
# #model1.most_similar('wind',topn=20)
#
# def findSimilar(model,word):
#     return(model.most_similar(word,topn=20))



'''Findout the nearest neighbors for all the words in the vocab'''
result={}
for word in vocab:
    result[word]=(findSimilar(model,word))


findSimilar(model,"mo")

'''Dump the result in the file'''
import json
fp=open('output_word2vec_meridium_assets.json','w')
fp.write(json.dumps(result))
fp.close()
##genrate the tsne plot


'''Under implementations...'''


from sklearn.manifold import MDS

#use this function to plot instead of tsne
def plotData(tfSparseMatrix,vocab):
    X=tfSparseMatrix.todense()
    pca = PCA(n_components=2).fit(X)
    #mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    data2D = pca.fit_transform(X)
    #plt.scatter(data2D[:,0], data2D[:,1])

    xs, ys = data2D[:, 0], data2D[:, 1]
    count=0
    for x, y, name in zip(xs, ys, vocab):
        plt.scatter(x, y)
        plt.text(x,y,vocab[count])
        count+=1

    # fig, ax = plt.subplots()
    #
    # for i, txt in enumerate(vocab):
    #     # print(i,txt)
    #     ax.annotate(txt, (data2D[i, 0], data2D[i, 1]))

    #plt.scatter(tt[:, 0], tt[:, 1])
    plt.show()


models=[]
#
# list(vocab).index("tc")
#
# models.append(model["tcs"])

indexes=[]
titles=[]
for i in range(150):
    models.append(model[vocab[i]])
    titles.append(vocab[i])


sparse_m = sp.sparse.bsr_matrix(np.array(models))

plotData(sparse_m,titles)

# generateTSNEPlot(vocab[1:5],model)


##tests
