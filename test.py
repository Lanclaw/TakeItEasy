import argparse
import pprint
import gensim
from glove import Glove
from glove import Corpus
#准备数据集
sentense = [['你','是','谁'],['我','是','中国人']]
corpus_model = Corpus()
corpus_model.fit(sentense, window=10)
#corpus_model.save('corpus.model')
print('Dict size: %s' % len(corpus_model.dictionary))
print('Collocations: %s' % corpus_model.matrix.nnz)

#训练
glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus_model.matrix, epochs=10,
          no_threads=1, verbose=True)
glove.add_dictionary(corpus_model.dictionary)

#模型保存
glove.save('glove.model')
glove = Glove.load('glove.model')
#语料保存
corpus_model.save('corpus.model')
corpus_model = Corpus.load('corpus.model')

#求相似词
glove.most_similar('我', number=10)