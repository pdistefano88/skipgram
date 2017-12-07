import tensorflow as tf
from lazy import lazy

class EmbeddingModel:

    def __init__(self, data, target, params):
        self.data = data
        self.target = target
        self.params = params
        self.embeddings
        self.cost
        self.optimize

    @lazy
    def embeddings(self):
        initial = tf.random_uniform(
            [self.params.vocabulary_size, self.params.embedding_size],
            -1.0, 1.0)
        return tf.Variable(initial)
    
    @lazy
    def optimize(self):
        optimizer = tf.train.AdamOptimizer()
        return optimizer.minimize(self.cost)

    @lazy
    def cost(self):
        embedded = tf.nn.embedding_lookup(self.embeddings, self.data)
        weight = tf.Variable(tf.truncated_normal(
            [self.params.vocabulary_size, self.params.embedding_size],
            stddev=1.0 / self.params.embedding_size ** 0.5))
        bias = tf.Variable(tf.zeros([self.params.vocabulary_size]))
        target = tf.expand_dims(self.target, 1)
        return tf.reduce_mean(tf.nn.nce_loss(
            weight, bias, target, embedded,
            self.params.contrastive_examples,
            self.params.vocabulary_size))   
