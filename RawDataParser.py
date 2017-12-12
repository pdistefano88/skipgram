import re
import os
import bz2
from helpers.download import download
from lxml import etree
import collections
import pickle


class Wikipedia:
    
    TOKEN_REGEX = re.compile(r'[A-Za-z]+|[!?.:,()]')

    def __init__(self, url, cache_dir, vocabulary_size=-1, max_count = 100, l = 0):
        self._cache_dir = os.path.expanduser(cache_dir)
        self._pages_path = os.path.join(self._cache_dir, 'pages.bz2')
        self._vocabulary_path = os.path.join(self._cache_dir, 'vocabulary.bz2')
        self._lambda = l
        if not os.path.isfile(self._pages_path):
            print('Read pages')
            self._read_pages(url)
        if not os.path.isfile(self._vocabulary_path):
            print('Build vocabulary')
            self._build_vocabulary()
        with open(os.path.join(self._cache_dir,"counter.pkl"), "rb") as f:
            self._counter = pickle.load(f)
        self._total_count = sum(self._counter.values())
        if vocabulary_size < 0:
            vocabulary_size = 1
            vocabulary_iter = iter(self._counter.most_common())
            while next(vocabulary_iter)[1] > max_count - 1:
                vocabulary_size += 1
        self._vocabulary_size = vocabulary_size            
        with bz2.open(self._vocabulary_path, 'rt') as vocabulary:
            print('Read vocabulary')
            self._vocabulary = [x.strip() for x, i in zip(vocabulary, range(self._total_count)) if i < self._vocabulary_size]
        self._indices = {x: i for i, x in enumerate(self._vocabulary)}

    def __iter__(self):
        """Iterate over pages represented as lists of word indices."""
        with bz2.open(self._pages_path, 'rt') as pages:
            for page in pages:
                words = page.strip().split()
                words = [self.encode(x) for x in words]
                yield words

    @property
    def vocabulary_size(self):
        return self._vocabulary_size
    
    @property
    def total_count(self):
        return self._total_count

    @property
    def counter(self):
        return self._counter


    def encode(self, word, limit_size = True):
        """Get the vocabulary index of a string word."""
        index = self._indices.get(word, 0)
        if index > self._vocabulary_size and limit_size == True:
            return 0
        else:
            return index

    def decode(self, index):
        """Get back the string word from a vocabulary index."""
        return self._vocabulary[index]
    
    def _read_pages(self, url):
        """
        Extract plain words from a Wikipedia dump and store them to the pages
        file. Each page will be a line with words separated by spaces.
        """
        wikipedia_path = download(url, self._cache_dir)
        with bz2.open(wikipedia_path) as wikipedia, \
                bz2.open(self._pages_path, 'wt') as pages:
            for _, element in etree.iterparse(wikipedia, tag='{*}page'):
                if element.find('./{*}redirect') is not None:
                    continue
                page = element.findtext('./{*}revision/{*}text')
                words = self._tokenize(page)
                pages.write(' '.join(words) + '\n')
                element.clear()

    def _build_vocabulary(self):
        """
        Count words in the pages file and write a list of the most frequent
        words to the vocabulary file.
        """
        counter = collections.Counter()
        with bz2.open(self._pages_path, 'rt') as pages:
            for page in pages:
                words = page.strip().split()
                counter.update(words)
        common = ['<unk>'] + counter.most_common()
        common = [x[0] for x in common]
        with bz2.open(self._vocabulary_path, 'wt') as vocabulary:
            for word in common:
                vocabulary.write(word + '\n')
        with open(os.path.join(self._cache_dir,"counter.pkl"), "wb") as f:
            pickle.dump(counter, f)    

    @classmethod
    def _tokenize(cls, page):
        words = cls.TOKEN_REGEX.findall(page)
        words = [x.lower() for x in words]
        return words
