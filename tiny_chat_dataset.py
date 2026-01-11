import torch
import json
from torch.utils.data import Dataset
from tokenizer import TiktokenTokenizer

class ChatDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_len=128, device="cpu"):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.eot_id = tokenizer.encoding.eot_token
        self.padding_id = -100
        self.device = device
        
        # Pre-process and move all data to the target device (e.g., GPU)
        self.X = []
        self.Y = []
        
        print(f"Pre-loading dataset to {device}...")
        for item in self.data:
            text = f"Instruction: {item['instruction']} "
            if item.get('input'):
                text += f"Input: {item['input']} "
            text += f"Output: {item['output']} <|endoftext|>"
            
            tokens = self.tokenizer.encode(text)
            
            if len(tokens) > self.max_len:
                tokens = tokens[:self.max_len]
            
            # Padding
            padding_len = self.max_len - len(tokens)
            padded_tokens_x = tokens + [self.eot_id] * padding_len
            padded_tokens_y = tokens + [self.padding_id] * padding_len
            
            self.X.append(torch.tensor(padded_tokens_x[:-1], device=self.device))
            self.Y.append(torch.tensor(padded_tokens_y[1:], device=self.device))
            
        # Convert list to a single tensor if possible (stacking)
        self.X = torch.stack(self.X)
        self.Y = torch.stack(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

if __name__ == "__main__":
    # Quick test
    tokenizer = TiktokenTokenizer()
    dataset = ChatDataset("tiny_chat_dataset.json", tokenizer)
    print(f"Dataset length: {len(dataset)}")
    
    x, y = dataset[0]
    print(f"X shape: {x.shape}")
    print(f"Y shape: {y.shape}")
    print(f"Sample X: {x[:10]}")
    print(f"Sample Y: {y[:10]}")