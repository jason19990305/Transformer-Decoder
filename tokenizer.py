
class WordTokenizer:
    def __init__(self , corpus):
        
        # Convert the corpus into a single string
        all_text = " ".join(corpus)

        # Split the string by space
        words = all_text.lower().split()

        # Remove duplicates and sort the words
        self.vocab = sorted(list(set(words)))

        # Create a dictionary to map words to indices
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}

        # Create a dictionary to map indices to words
        self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}
        
        self.vocab_size = len(self.vocab)
    
    def encode(self, text)->list[int]:
        # Convert the text into a list of indices
        return [self.word_to_idx.get(word, 0) for word in text.lower().split()] 
    
    def decode(self, indices)->str:
        # Convert the list of indices into a string of text
        return " ".join([self.idx_to_word.get(idx, "") for idx in indices])

if __name__ == "__main__":
    corpus = [
        "red apple is sweet",
        "blue sky is high",
        "red fire is hot",
        "blue sea is deep"
    ]
    tokenizer = WordTokenizer(corpus)
    print("Vocabulary:", tokenizer.vocab)
    print("Word to Index:", tokenizer.word_to_idx)
    print("Index to Word:", tokenizer.idx_to_word)
    print("Vocabulary Size:", tokenizer.vocab_size)
    print("Encoded Text blue sky:", tokenizer.encode("blue sky"))
    print("Decoded Text 4,5:", tokenizer.decode([4, 5]))