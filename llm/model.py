# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionHead(nn.Module):
    def __init__(self, embedding_dim, head_size, sequence_length):
        super().__init__()
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)

        self.attention_mask = torch.tril(torch.ones(sequence_length, sequence_length))
        self.dropout = nn.Dropout(0.1)
        self.scaling_factor = head_size ** -0.5

    def forward(self, token_embeddings):
        batch_size, sequence_length, embedding_dim = token_embeddings.shape

        queries = self.query(token_embeddings)  # (B, T, head_size)
        keys = self.key(token_embeddings)       # (B, T, head_size)
        values = self.value(token_embeddings)   # (B, T, head_size)

        attention_scores = (queries @ keys.transpose(-2, -1)) * self.scaling_factor
        attention_scores = attention_scores.masked_fill(
            self.attention_mask[:sequence_length, :sequence_length] == 0, float('-inf')
        )

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = attention_weights @ values  # (B, T, head_size)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, number_of_heads, sequence_length):
        super().__init__()
        head_size = embedding_dim // number_of_heads
        self.attention_heads = nn.ModuleList([
            SelfAttentionHead(embedding_dim, head_size, sequence_length) for _ in range(number_of_heads)
        ])
        self.linear_projection = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, token_embeddings):
        concatenated_outputs = torch.cat(
            [head(token_embeddings) for head in self.attention_heads], dim=-1
        )
        projected_output = self.linear_projection(concatenated_outputs)
        return self.dropout(projected_output)


class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.linear2 = nn.Linear(4 * embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.dropout(self.linear2(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, number_of_heads, sequence_length):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.multi_head_attention = MultiHeadAttention(embedding_dim, number_of_heads, sequence_length)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForwardNetwork(embedding_dim)

    def forward(self, token_embeddings):
        x = token_embeddings + self.multi_head_attention(self.layer_norm_1(token_embeddings))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class GPTMiniModel(nn.Module):
    def __init__(self, vocabulary_size, sequence_length, embedding_dim=128, number_of_heads=4, number_of_layers=4):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocabulary_size, embedding_dim)
        self.position_embedding_table = nn.Embedding(sequence_length, embedding_dim)

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embedding_dim, number_of_heads, sequence_length) for _ in range(number_of_layers)]
        )

        self.final_layer_norm = nn.LayerNorm(embedding_dim)
        self.language_model_head = nn.Linear(embedding_dim, vocabulary_size)

    def forward(self, input_indices, target_indices=None):
        batch_size, sequence_length = input_indices.shape
        token_embeddings = self.token_embedding_table(input_indices)
        position_indices = torch.arange(sequence_length, device=input_indices.device)
        position_embeddings = self.position_embedding_table(position_indices)

        x = token_embeddings + position_embeddings
        x = self.transformer_blocks(x)
        x = self.final_layer_norm(x)
        logits = self.language_model_head(x)

        if target_indices is None:
            return logits

        # reshape logits and targets for loss computation
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.view(batch_size * seq_len, vocab_size)
        target_indices = target_indices.view(batch_size * seq_len)
        loss = F.cross_entropy(logits, target_indices)
        return logits, loss
