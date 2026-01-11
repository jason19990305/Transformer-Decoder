import torch
import torch.nn as nn
from tokenizer import WordTokenizer

class WordEmbedding(nn.Module):
    def __init__(self, num_vocab, embedding_size):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_vocab, embedding_size)
    
    def forward(self, x):
        return self.embedding(x)
        
if __name__ == "__main__":
    corpus = [
        "red apple is sweet",
        "blue sky is high",
        "red fire is hot",
        "blue sea is deep"
    ]
    tokenizer = WordTokenizer(corpus)
    num_vocab = tokenizer.vocab_size
    embedding_size = 3
    word_embedding = WordEmbedding(num_vocab, embedding_size)
    encoded_text = tokenizer.encode("red apple")
    print(encoded_text)
    print(word_embedding(torch.tensor(encoded_text)))