import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embedding_size = embedding_size
        self.max_len = max_len

        # Create a matrix of shape (max_len, embedding_size)
        pe = torch.zeros(max_len, embedding_size)

        # Calculate the positional encoding 0~max_len
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)    

        # Calculate the divisor of the positional encoding
        # Formula: 10000^(2i/d)
        div_term = torch.exp(torch.arange(0, embedding_size, 2) *
         (-torch.log(torch.tensor(10000.0)) / embedding_size))

        # Calculate the positional encoding
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Reshape the positional encoding
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    
    def forward(self, t)->torch.Tensor:
        # t is an integer index
        # return the positional encoding for the given position t
        return self.pe[:, t, :]

def visualize_pe(pe_matrix, max_len=100, embedding_size=32):

    # Use only the specified range for better visualization
    # pe_matrix shape is (1, 5000, 32)
    data = pe_matrix[0, :max_len, :].T.cpu().numpy() # Shape: (embedding_size, max_len)
    
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(data, cmap='RdBu', vmin=-1, vmax=1)
    plt.xlabel('Position (X-axis)')
    plt.ylabel('Encoding Vector Dimensions (Y-axis)')
    plt.title('Positional Encoding Heatmap')
    plt.colorbar(label='Value (-1 to 1)')
    
    # Save the figure
    plt.savefig('positional_encoding_heatmap.png')
    print("Heatmap saved as 'positional_encoding_heatmap.png'")
    # Note: Depending on your environment, plt.show() might not work in some terminals.
    plt.show()

if __name__ == "__main__":
    embedding_size = 32
    max_len = 100 # Reduced for plotting
    pe_layer = PositionalEncoding(embedding_size, max_len)
    
    print(f"PE Shape: {pe_layer.pe.shape}")
    
    # 1. Test lookup (fetch position 1)
    pos_1_vector = pe_layer(1)
    print(f"Vector at position 1:\n{pos_1_vector}")
    
    # 2. Visualize
    visualize_pe(pe_layer.pe, max_len=max_len, embedding_size=embedding_size)