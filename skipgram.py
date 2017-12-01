
# coding: utf-8

# In[1]:

import bz2
import collections
import os
import tensorflow as tf
import numpy as np
#import lazy_property
from helpers import download
from lxml import etree
from attrdict import AttrDict
from EmbeddingModel import EmbeddingModel
from RawDataParser import Wikipedia
from withableWriter import withableWriter


# In[2]:

def skipgrams(pages, max_context):
    """Form training pairs according to the skip-gram model."""
    for words in pages:
        for index, current in enumerate(words):
            context = np.random.randint(1, max_context)
            for target in words[max(0, index - context): index]:
                yield current, target
            for target in words[index + 1: index + context + 1]:
                yield current, target


def batched(iterator, batch_size):
    """Group a numerical stream into batches and yield them as Numpy arrays."""
    while True:
        data = np.zeros(batch_size)
        target = np.zeros(batch_size)
        for index in range(batch_size):
            data[index], target[index] = next(iterator)
        yield data, target
    
params = AttrDict(
    vocabulary_size=10000,
    max_context=10,
    embedding_size=200,
    contrastive_examples=100,
    learning_rate=0.5,
    momentum=0.5,
    batch_size=1000,
)  

data = tf.placeholder(tf.int32, [None])
target = tf.placeholder(tf.int32, [None])
model = EmbeddingModel(data, target, params)

corpus = Wikipedia(
    'https://dumps.wikimedia.org/enwiki/20171120/enwiki-20171120-pages-meta-current1.xml-p10p30303.bz2',
    '~/wikipedia',
    params.vocabulary_size)
examples = skipgrams(corpus, params.max_context)
batches = batched(examples, params.batch_size)

cost_summary = tf.summary.scalar(tensor = model.cost, name = 'cost')

sess = tf.Session()
sess.run(tf.global_variables_initializer())
average = collections.deque(maxlen=100)
for index, batch in enumerate(batches):
    feed_dict = {data: batch[0], target: batch[1]}
    cost, _ = sess.run([model.cost, model.optimize], feed_dict)
    average.append(cost)
    print('{}: {:5.1f}'.format(index + 1, sum(average) / len(average)))

embeddings = sess.run(model.embeddings)
np.save('/home/pietro/wikipedia/embeddings.npy', embeddings)

