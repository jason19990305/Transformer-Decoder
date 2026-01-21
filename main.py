import torch
import torch.nn as nn
from transformer import Model
from tokenizer import TiktokenTokenizer
from tiny_chat_dataset import ChatDataset
from torch.utils.data import DataLoader
import time

def count_parameters(model):
    """
    Count the total and trainable parameters of a model, and estimate its memory size.
    Assuming Float32 (4 bytes per parameter).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = (total_params * 4) / (1024 * 1024)
    
    print("-" * 30)
    print(f"Model Parameter Statistics:")
    print(f"  Total Parameters     : {total_params:,}")
    print(f"  Trainable Parameters : {trainable_params:,}")
    print(f"  Estimated Model Size : {model_size_mb:.2f} MB")
    print("-" * 30)
    return total_params, trainable_params, model_size_mb

if __name__ == "__main__":
    # Settings
    json_path = "tiny_chat_dataset.json"
    batch_size = 20
    embedding_size = 128
    num_layers = 4
    num_epochs = 100
    lr = 0.0005
    max_seq_len = 128
    
    # Initialize Tokenizer and Model's vocabulary size
    tokenizer = TiktokenTokenizer("gpt-4o")
    num_vocab = tokenizer.vocab_size

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Device name : ", torch.cuda.get_device_name(0))

    # Prepare Dataset and DataLoader
    dataset = ChatDataset(json_path, tokenizer, max_len=max_seq_len, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model , Loss and Optimizer
    model = Model(num_vocab, embedding_size, num_layers).to(device)
    
    # Print parameter statistics
    count_parameters(model)
    
    # Ignore the padding token in loss calculation
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start = time.time()
    # Training Loop
    print(f"Starting training on {len(dataset)} samples (all-on-GPU enabled)...")
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, targets in dataloader:
            # Inputs and targets are already on device
            optimizer.zero_grad()
            output = model(inputs)
            
            # Loss expects (batch, vocab, seq_len)
            loss = criterion(output.transpose(1, 2), targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        if (epoch+1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}')
            
    end = time.time()
    print(f"Training time : {end - start:.2f} seconds")

    # Inference / Testing
    print("\n=== Chat Testing ===")
    model.eval()
    test_prompts = [
        "Instruction: 你好，請自我介紹一下。 Output:",
        "Instruction: 1 加 1 等於多少？ Output:",
        "Instruction: 請將下面的英文翻譯成繁體中文。 Input: Deep Learning is fascinating. Output:",
        "Instruction: 再見的日文怎麼說？ Output:"
    ]
    
    eot_id = tokenizer.encoding.eot_token
    
    for prompt in test_prompts:
        print(f"Prompt: {prompt}")
        current_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
        
        # Generation Loop
        with torch.no_grad():
            for _ in range(100):
                output = model(current_ids)
                # Get prediction for the last token only
                last_token_logits = output[0, -1, :]
                predicted_id = torch.argmax(last_token_logits).item()
                
                # Append predicted token to sequence
                next_id = torch.tensor([[predicted_id]]).to(device)
                current_ids = torch.cat([current_ids, next_id], dim=1)
                
                if predicted_id == eot_id:
                    break
            
        generated_text = tokenizer.decode(current_ids[0].tolist())
        print(f"Generated: {generated_text}")
        print("-" * 20)