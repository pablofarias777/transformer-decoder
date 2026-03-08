import numpy as np


def cross_attention(encoder_out, decoder_state):

    d_model = encoder_out.shape[-1]

    Wq = np.random.randn(d_model, d_model)
    Wk = np.random.randn(d_model, d_model)
    Wv = np.random.randn(d_model, d_model)

    Q = decoder_state @ Wq
    K = encoder_out @ Wk
    V = encoder_out @ Wv

    scores = Q @ K.transpose(0,2,1)

    scores = scores / np.sqrt(d_model)

    exp_scores = np.exp(scores - np.max(scores))
    attention = exp_scores / np.sum(exp_scores)

    output = attention @ V
    
    return output