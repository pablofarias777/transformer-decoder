import numpy as np
from mask import create_causal_mask

seq_len = 5

mask = create_causal_mask(seq_len)

print("Causal Mask:")
print(mask)