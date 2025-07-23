from tokenizer import CharTokenizer

sample_text  = 'Hello LLM'

tokenizer = CharTokenizer(sample_text)

encode = tokenizer.encode(sample_text)
print(f'Encoded: {encode}')
decode = tokenizer.decode(encode)
print(f'Decoded: {decode}')

# Check correctness
assert decode == sample_text, "Decoded text does not match original text."