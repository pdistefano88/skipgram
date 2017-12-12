import os
import tensorflow as tf
import numpy as np
import collections
from attrdict import AttrDict
from EmbeddingModel import EmbeddingModel
from RawDataParser import Wikipedia
from withableWriter import withableWriter
from runManager import getRun

def subsample(frequency, l):
    r = np.random.uniform()
    return(r > 1 - np.sqrt(l/frequency))

def skipgrams(pages, max_context, l):
    """Form training pairs according to the skip-gram model."""
    for words in pages:
        for index, current in enumerate(words):
            if current != 0 and not subsample(pages.counter[pages.decode(current)]/pages.total_count, l):
                continue
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
    embedding_size=300,
    contrastive_examples=15,
    learning_rate=0.025,
    momentum=0.9,
    batch_size=24,
    num_epochs = 2,
    l = 0.00005
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


examples = skipgrams(corpus, params.max_context, params.l)
batches = batched(examples, params.batch_size)

average = collections.deque(maxlen=100)
tf_average = tf.Variable(0.)
cost_summary = tf.summary.scalar(name = 'cost', tensor = tf_average)
merged_summaries = tf.summary.merge_all()

directory = getRun("wikipedia")

print("run in " + directory)

save_step = 1000000

with tf.Session() as sess, withableWriter(directory) as writer:
    sess.run(tf.global_variables_initializer())
    step = 0
    for index, batch in enumerate(batches):
        feed_dict = {data: batch[0], target: batch[1]}
        cost, _, ms = sess.run([model.cost, model.optimize, merged_summaries], feed_dict)
        average.append(cost)
        if step % 10000 == 0:
            sess.run(tf_average.assign(sum(average) / len(average)))
            writer.add_summary(ms, global_step=step)
            writer.flush()
            print('{}: {:5.1f}'.format(index + 1, sum(average) / len(average)))            
        if step % save_step == 0:
            np.save('embeddings_' + str(params.vocabulary_size) + '_' + str(step) +'.npy', sess.run(model.embeddings))
            try:
                os.remove('embeddings_' + str(params.vocabulary_size) + '_' + str(step - int(10 * save_step)) + '.npy')
            except:
                pass
        step +=1
