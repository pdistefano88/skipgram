import tensorflow as tf

class withableWriter:
    ##Opens a tf.summary.FileWriter at the beginning of a with clause and closes it the end 
    def __init__(self, directory, graph = tf.get_default_graph()):
        self.directory = directory
        self.graph = graph
        
    def __enter__(self):
        self.writer = tf.summary.FileWriter(self.directory)
        self.writer.flush()

        return(self.writer)
    
    def __exit__(self, type, value, traceback):
        self.writer.close()

