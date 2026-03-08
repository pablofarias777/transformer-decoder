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