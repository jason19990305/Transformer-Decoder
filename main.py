import torch
import torch.nn as nn
from transformer import Model
from tokenizer import WordTokenizer
import time

if __name__ == "__main__":
    corpus = [
        "red apple is sweet <eos>",
        "blue sky is high <eos>",
        "red fire is hot <eos>",
        "blue sea is deep <eos>"
    ]
    corpus_length = len(corpus)
    # Tokenize the corpus
    tokenizer = WordTokenizer(corpus)
    # Get the number of vocabulary
    num_vocab = tokenizer.vocab_size
    # Set the embedding size = 16 for ploting purpose
    embedding_size = 16

    # Prepare Data
    inputs = []
    targets = []
    for sentence in corpus:
        tokens = tokenizer.encode(sentence)
        inputs.append(tokens[:-1])
        targets.append(tokens[1:])

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    if device.type == "cuda":
        print("Device name : ", torch.cuda.get_device_name(0))
    
    # inputs shape : (batch_size, seq_len)
    inputs = torch.tensor(inputs).to(device)
    # targets shape : (batch_size, seq_len)
    targets = torch.tensor(targets).to(device)
    print("Input shape : ", inputs.shape)
    print("Target shape : ", targets.shape)

    # Hyperparameters
    num_epochs = 1000
    lr = 0.001
    num_layers = 4

    # Model , Loss and Optimizer
    model = Model(num_vocab, embedding_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start = time.time()
    # Training
    for epoch in range(num_epochs):       
        optimizer.zero_grad()
        # output : [batch_size, seq_len, num_vocab]
        output = model(inputs)
        # targets : [batch_size, seq_len]
        # output Transpose : [batch_size, num_vocab, seq_len]
        # Alignment least dimension
        loss = criterion(output.transpose(1, 2), targets)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    end = time.time()
    print(f"Training time : {end - start:.2f} seconds")


    # Testing
    print("\n=== Testing ===")
    test_sentences = ["red apple", "red fire","blue","red"]
    
    # Get EOS ID
    eos_id = tokenizer.word_to_idx.get("<eos>")
    
    for sentence in test_sentences:
        print(f"Input: {sentence}")
        # Encode
        # current_ids : [1, seq_len]
        current_ids = torch.tensor([tokenizer.encode(sentence)]).to(device)
        
        # Predict until <eos> or max_len
        for _ in range(20):
            # output : [1, seq_len, num_vocab]
            output = model(current_ids)
            # Get the last token prediction
            last_token_logits = output[0, -1, :]
            predicted_id = torch.argmax(last_token_logits).item()
            
            # Append to inputs
            current_ids = torch.cat([current_ids, torch.tensor([[predicted_id]]).to(device)], dim=1)
            
            if predicted_id == eos_id:
                break
            
        # Decode
        generated_text = tokenizer.decode(current_ids[0].tolist())
        print(f"Generated: {generated_text}")
        print("-" * 20)