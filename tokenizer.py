import tiktoken

class TiktokenTokenizer:
    def __init__(self, model_name="gpt-4o"):
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("o200k_base")
        
        self.vocab_size = self.encoding.n_vocab

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text, allowed_special="all")

    def decode(self, ids: list[int]) -> str:
        return self.encoding.decode(ids)

def usage_example():
    # 1. Automatically load the encoder for the corresponding model
    tokenizer = TiktokenTokenizer("gpt-4o")
    print(f"Using encoder: {tokenizer.encoding.name}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # 2. Prepare test text (includes English and Emoji)
    text = "Hello, world!🚀"

    # 3. Encode (Text -> Token IDs)
    token_ids = tokenizer.encode(text)
    print(f"\nOriginal text: {text}")
    print(f"Token IDs: {token_ids}")
    print(f"Token count: {len(token_ids)}")

    # 4. Decode (Token IDs -> Text)
    decoded_text = tokenizer.decode(token_ids)
    print(f"Decoded text: {decoded_text}")

    # 5. Inspect each ID and its corresponding bytes (Visualization)
    print("\n--- Token Segmentation Details ---")
    for tid in token_ids:
        token_bytes = tokenizer.encoding.decode_single_token_bytes(tid)
        try:
            print(f"ID: {tid:<10} | Content: {token_bytes.decode('utf-8')}")
        except UnicodeDecodeError:
            print(f"ID: {tid:<10} | Content: {token_bytes}")

if __name__ == "__main__":
    usage_example()