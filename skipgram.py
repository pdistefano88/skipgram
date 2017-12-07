import os
import tensorflow as tf
import numpy as np
import collections
from attrdict import AttrDict
from EmbeddingModel import EmbeddingModel
from RawDataParser import Wikipedia
from withableWriter import withableWriter
from runManager import getRun

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
    vocabulary_size=-1,
    max_context=10,
    embedding_size=500,
    contrastive_examples=10,
    learning_rate=0.01,
    momentum=0.9,
    batch_size=20,
)  


corpus = Wikipedia(
    'https://dumps.wikimedia.org/enwiki/20171120/enwiki-20171120-pages-articles-multistream.xml.bz2',
    'wikipedia',
    params.vocabulary_size)

print("total number of words: %i" % corpus.total_count)
total_steps =  params.max_context / 2 * corpus.total_count / params.batch_size
print("steps in one epoch: %f" % total_steps)
print("vocabulary size: %i" % corpus.vocabulary_size)

params.vocabulary_size = corpus.vocabulary_size
data = tf.placeholder(tf.int32, [None])
target = tf.placeholder(tf.int32, [None])
model = EmbeddingModel(data, target, params)


examples = skipgrams(corpus, params.max_context)
batches = batched(examples, params.batch_size)

cost_summary = tf.summary.scalar(name = 'cost', tensor = tf.reduce_mean(model.cost))
merged_summaries = tf.summary.merge_all()

directory = getRun("wikipedia")

print("run in " + directory)

with tf.Session() as sess, withableWriter(directory) as writer:
    sess.run(tf.global_variables_initializer())
    average = collections.deque(maxlen=100)
    step = 0
    for index, batch in enumerate(batches):
        feed_dict = {data: batch[0], target: batch[1]}
        cost, _, ms = sess.run([model.cost, model.optimize, merged_summaries], feed_dict)
        average.append(cost)
        writer.add_summary(ms, global_step=step)        
        if step % 1000 == 999:
            writer.flush()
            print('{}: {:5.1f}'.format(index + 1, sum(average) / len(average)))            
        if step % 1000000 == 999999:
            np.save('embeddings_' + str(params.vocabulary_size) + '_' + str(step) +'.npy', sess.run(model.embeddings))
            try:
                os.remove('embeddings_' + str(params.vocabulary_size) + '_' + str(step - 10000000) +'.npy')
            except:
                pass
        step +=1