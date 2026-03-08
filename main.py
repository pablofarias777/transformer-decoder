import numpy as np
from mask import create_causal_mask

seq_len = 5

mask = create_causal_mask(seq_len)

print("Causal Mask:")
print(mask)

from cross_attention import cross_attention

encoder_output = np.random.randn(1,10,512)

decoder_state = np.random.randn(1,4,512)

cross_out = cross_attention(encoder_output, decoder_state)

print("\nCross Attention Output Shape:")
print(cross_out.shape)

def generate_next_token(current_sequence, encoder_out):

    vocab_size = 10000

    logits = np.random.randn(vocab_size)

    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    return probs

    print("\nSimulando geração de texto:")
    
current_sequence = ["<START>"]

encoder_out = np.random.randn(1,10,512)

vocab = ["O", "rato", "comeu", "queijo", "<EOS>"]

while True:

    probs = generate_next_token(current_sequence, encoder_out)

    next_token_id = np.argmax(probs)

    next_token = vocab[next_token_id % len(vocab)]

    current_sequence.append(next_token)

    print(current_sequence)

    if next_token == "<EOS>":
        break