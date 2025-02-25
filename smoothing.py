from collections import Counter, defaultdict
# This file made by Rohan for Smoothing and Unknown Words Code - integrated into main

class NGramModel:
    def __init__(self, words, k=1e-5):
        """
        Initialize the n-gram model with Laplace/Add-k smoothing.
        :param words: List of tokens from the dataset.
        :param k: Smoothing parameter set to 1e-5
        """
        self.k = k
        self.words = words
        self.unigram_counts = Counter(words)
        self.bigram_counts = Counter((words[i], words[i+1]) for i in range(len(words)-1))
        self.vocab = set(words)
        self.total_unigrams = sum(self.unigram_counts.values())
        self.total_bigrams = sum(self.bigram_counts.values())

    def handle_unknown_words(self, token_list):
        #Here we will replace unseen words with <UNK>
        return [word if word in self.vocab else "<UNK>" for word in token_list]

    def calculate_smoothed_unigram_probabilities(self):
        #Here we compute the unigram probabilities with laplace smoothing.
        vocab_size = len(self.vocab)
        return {
            word: (self.unigram_counts[word] + self.k) / (self.total_unigrams + self.k * vocab_size)
            for word in self.vocab
        }

    def calculate_smoothed_bigram_probabilities(self):
        #This section computed bigram probabilities with add-k smoothing.
        vocab_size = len(self.vocab)
        smoothed_bigrams = defaultdict(lambda: self.k / (self.total_unigrams + self.k * vocab_size))

        for w1 in self.vocab:
            counts = self.unigram_counts[w1] + self.k * vocab_size
            for w2 in self.vocab:
                if (w1, w2) in self.bigram_counts:
                    smoothed_bigrams[(w1, w2)] = (self.bigram_counts[(w1, w2)] + self.k) / counts
                else:
                    smoothed_bigrams[(w1, w2)] = self.k / counts

        return smoothed_bigrams

        return smoothed_bigrams
