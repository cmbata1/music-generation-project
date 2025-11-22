import torch
import torch.nn as nn

class MusicGRUModel(nn.Module):
    """
    Embedding -> GRU -> Linear
    Input:  (batch_size, seq_len) of token IDs
    Output: (batch_size, vocab_size) logits for the next token
    """
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128,
                 num_layers=1, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
        x: (batch_size, seq_len) of token IDs
        """
        emb = self.embedding(x)             
        out, h_n = self.gru(emb)           
        last_hidden = out[:, -1, :]   
        last_hidden = self.dropout_layer(last_hidden)     
        logits = self.fc(last_hidden)      
        return logits
    
def load_trained_model(checkpoint_path, device):
    """
    Load a trained MusicGRUModel from a checkpoint.
    Returns: (model, seq_len)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    vocab_size = int(checkpoint["vocab_size"])
    embed_dim = int(checkpoint["embed_dim"])
    hidden_dim = int(checkpoint["hidden_dim"])
    num_layers = int(checkpoint["num_layers"])
    dropout = float(checkpoint["dropout"])
    seq_len = int(checkpoint["seq_len"])

    model = MusicGRUModel(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, seq_len