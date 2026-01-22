import torch
import torch.nn as nn
from word_embedding import WordEmbedding
from positional_encoding import PositionalEncoding

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, n_heads=8, max_len=5000):
        super(MaskedMultiHeadAttention, self).__init__()
        self.embedding_size = embedding_size
        self.n_heads = n_heads
        
        # Verify that embedding_size is divisible by n_heads
        assert embedding_size % n_heads == 0, "embedding_size must be divisible by n_heads"
        
        self.head_dim = embedding_size // n_heads
        
        # Initialize the weight matrix for the positional encoding
        self.W_q = nn.Linear(embedding_size, embedding_size, bias=False)
        self.W_k = nn.Linear(embedding_size, embedding_size, bias=False)
        self.W_v = nn.Linear(embedding_size, embedding_size, bias=False)
        # > Note : nn.Linear is like a matrix multiplication
        
        # Output Linear projection
        self.fc_out = nn.Linear(embedding_size, embedding_size)

        # Causal mask : to mask out the future tokens
        # Shape is (max_len, max_len)
        # [1, 0, 0, 0]
        # [1, 1, 0, 0]
        # [1, 1, 1, 0]
        # [1, 1, 1, 1]
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len)))

    def forward(self, x):   
        # x shape is (batch_size, seq_len, embedding_size)
        # return shape is (batch_size, seq_len, embedding_size)

        batch_size, seq_len, embedding_size = x.shape
        
        # Compute Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        # > Note : Q, K, V shape is (batch_size, seq_len, embedding_size)
        
        # Reshape Q, K, V for Multi-Head Attention
        # Split embedding_size into n_heads * head_dim
        # (batch_size, seq_len, embedding_size) -> (batch_size, seq_len, n_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Transpose for attention calculation: (batch_size, n_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute the attention scores
        # (batch_size, n_heads, seq_len, head_dim) * (batch_size, n_heads, head_dim, seq_len) 
        # -> (batch_size, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # mask : to mask out the future tokens
        # mask shape (seq_len, seq_len) broadcasted to (batch_size, n_heads, seq_len, seq_len)
        mask = self.mask[:seq_len, :seq_len]
        # find the position where mask is 0 and fill it with -inf
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Compute the attention weights
        attention_weights = torch.softmax(scores, dim=-1)
        # > Note : attention_weights shape is (batch_size, n_heads, seq_len, seq_len)

        # Compute the output
        # (batch_size, n_heads, seq_len, seq_len) * (batch_size, n_heads, seq_len, head_dim)
        # -> (batch_size, n_heads, seq_len, head_dim)
        output = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        # (batch_size, n_heads, seq_len, head_dim) -> (batch_size, seq_len, n_heads, head_dim)
        output = output.transpose(1, 2).contiguous()
        # -> (batch_size, seq_len, embedding_size)
        output = output.view(batch_size, seq_len, self.embedding_size)
        
        # Final linear projection
        output = self.fc_out(output)
        

        return output

class TransformerDecoder(nn.Module):
    def __init__(self, embedding_size):
        super(TransformerDecoder, self).__init__()

        # 1. Masked Multi-Head Self-Attention
        self.multi_head_attention = MaskedMultiHeadAttention(embedding_size)
        
        # 2. Norm layers (RMSNorm)
        # Using Pytorch built-in RMSNorm
        self.norm1 = nn.RMSNorm(embedding_size)
        self.norm2 = nn.RMSNorm(embedding_size)

        # 3. Residual Dropouts
        # These correspond to 'resid_pdrop' in GPT configs.
        self.resid_drop1 = nn.Dropout(0.1)
        self.resid_drop2 = nn.Dropout(0.1)
        
        # MLP (Feed Forward)
        # Expansion factor is usually 4 in vanilla Transformer
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * 4),
            nn.ReLU(),
            nn.Linear(embedding_size * 4, embedding_size),
            nn.Dropout(0.1) # Internal dropout for the MLP
        )
        
    
    def forward(self, x):
        # === Part 1: Attention Block ===
        # Pre-Norm: Normalize -> Attention -> Dropout -> Add Residual
        residual = x
        x = self.norm1(x) 
        x = self.multi_head_attention(x)
        x = self.resid_drop1(x)
        x = x + residual  # Residual Connection

        # === Part 2: Feed-Forward Block (MLP) ===
        # Pre-Norm: Normalize -> MLP -> Dropout -> Add Residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.resid_drop2(x)
        x = x + residual  # Residual Connection

        return x

class Model(nn.Module):
    def __init__(self, num_vocab, embedding_size, num_layers=4):
        super(Model, self).__init__()
        self.word_embedding = WordEmbedding(num_vocab, embedding_size)
        self.positional_encoding = PositionalEncoding(embedding_size)
        
        self.decoders = nn.ModuleList([
            TransformerDecoder(embedding_size) for _ in range(num_layers)
        ])
        
        self.linear = nn.Linear(embedding_size, num_vocab)

    def forward(self, x):
        # 1. Embedding + position_encoding 
        embedding = self.word_embedding(x)
        
        # Create positions tensor [0, 1, 2, ..., seq_len-1]
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        positional_encoding = self.positional_encoding(positions)
        
        x = embedding + positional_encoding
        
        # Pass through each decoder layer
        for decoder in self.decoders:
            x = decoder(x)
            
        x = self.linear(x)
        return x

if __name__ == "__main__":
    num_vocab = 100
    embedding_size = 16
    batch_size = 2
    seq_len = 5
    num_layers = 2
    
    model = Model(num_vocab, embedding_size, num_layers=num_layers)
    
    # Create dummy input
    x = torch.randint(0, num_vocab, (batch_size, seq_len))
    
    print(f"Input shape: {x.shape}")
    
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    expected_shape = (batch_size, seq_len, num_vocab)
    if output.shape == expected_shape:
        print("Test PASSED: Output shape is correct.")
    else:
        print(f"Test FAILED: Expected shape {expected_shape}, but got {output.shape}")
