import re
import math
from collections import Counter

def text_preprocessing(content):
    """
    Perform text preprocessing including:
    - Lowercasing
    - Extracting word tokens using regex
    """
    return re.findall(r"\b\w+\b", content.lower())

def calculate_unigram_probabilities(words):
    """
    Compute unigram probabilities without smoothing.
    """
    word_counts = Counter(words)
    total_words = sum(word_counts.values())
    return {word: freq / total_words for word, freq in word_counts.items()}

def calculate_bigram_probabilities(words):
    """
    Compute bigram probabilities without smoothing.
    """
    unigram_counts = Counter(words)
    bigram_counts = Counter((words[i], words[i+1]) for i in range(len(words)-1))
    return {bigram: count / unigram_counts[bigram[0]] for bigram, count in bigram_counts.items()}

def calculate_uni_perplexity(words, probList, smoothed_model):
    probSum = 0.0
    word_count = 0
    words = smoothed_model.handle_unknown_words(words)

    for word in words:
        prob = probList.get(word)
        probSum += math.log2(prob)
        word_count += 1

    avgLogProb = probSum / word_count
    perplexity = 2 ** (- avgLogProb)

    return perplexity

def calculate_bi_perplexity(words, probList, smoothed_model):
    probSum = 0.0
    word_count = 0
    words = smoothed_model.handle_unknown_words(words)

    for i in range(len(words) - 1):
        bigram = (words[i], words[i + 1])
        prob = probList.get(bigram)
        probSum += math.log2(prob)
        word_count += 1

    avgLogProb = probSum / word_count
    perplexity = 2 ** (- avgLogProb)

    return perplexity

from smoothing import NGramModel

def main():
    filename = "train.txt"
    
    with open(filename, "r", encoding="utf-8") as f:
        data = f.read()
    
    tokens = text_preprocessing(data)
    tokens.append("<UNK>")

    # Initialize smoothed model
    smoothed_model = NGramModel(tokens, k=0.1)
    
    # Compute smoothed n-gram probabilities
    smoothed_unigram_probs = smoothed_model.calculate_smoothed_unigram_probabilities()
    smoothed_bigram_probs = smoothed_model.calculate_smoothed_bigram_probabilities()
    
    print("Smoothed Unigram Probabilities:")
    for word, prob in smoothed_unigram_probs.items():
        print(f"P({word}) = {prob:.4f}")
    
    print("\nSmoothed Bigram Probabilities:")
    for bigram, prob in smoothed_bigram_probs.items():
        print(f"P({bigram[1]}|{bigram[0]}) = {prob:.4f}")

    filename = "val.txt"

    with open(filename, "r", encoding="utf-8") as f:
        data = f.read()

    tokens = text_preprocessing(data)

    unigram_perplexity = calculate_uni_perplexity(tokens, smoothed_unigram_probs, smoothed_model)
    bigram_perplexity = calculate_bi_perplexity(tokens, smoothed_bigram_probs, smoothed_model)

    print(f"\nUnigram Perplexity: {unigram_perplexity:.4f}\nBigram Perplexity: {bigram_perplexity:.4f}")

if __name__ == "__main__":
    main()
