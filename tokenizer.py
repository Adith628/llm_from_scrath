class CharTokenizer:
    def __init__(self,text):
        self.chars = sorted(list(set(text)))
        self.stoi = {ch:i for i,ch in enumerate(self.chars)}
        self.itos = {i:ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.chars)
        
    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, tokens):
        return ''.join([self.itos[i] for i in tokens])
