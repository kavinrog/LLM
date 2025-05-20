import numpy as np 

def softmax(x):
    e_x = np.exp(x - np.max(x, axis = 1, keepdims= True))
    return e_x / np.sum(axis = 1, keepdims= True)

def layernorm(x, epsilon=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x-mean) / np.sqrt(var + epsilon)

def scaled_dot_product_attention(query, key, value, mask = None):
    d_k = query.shape[-1]
    scores = np.matmul(query, key.transpose(0, 1, 3, 2))/np.sqrt(d_k)
    if mask is not None:
        scores += (mask * 1e-9)
    attn = softmax(scores)
    output = attn @ value
    return output

def multihead_attention(x, num_heads = 8):
    batch_size, seq_len, d_model = x.shape
    d_k = d_model // num_heads
    
    def split_heads(x):
        x = x.reshape(batch_size, seq_len, num_heads, d_k)
        return x.transpose(0, 2, 1, 3)
    Q = split_heads(x)
    K = split_heads(x)
    V = split_heads(x)
    attn = scaled_dot_product_attention(Q, K, V)
    attn = attn.transpose(0, 2, 1, 3)
    attn = attn.reshape(batch_size, seq_len, d_model)
    return attn

def feed_forward(x, d_ff = 2048):
    W1 = np.random.rand(x.shape[-1], dff)
    b1 = np.random.rand(d_ff)
    W2 = np.random.rand(d_ff, x.shape[-1])
    b2 = np.random.rand(x.shape[-1])
    x = np.matmul(x, W1) + b1
    x = np.maximum(0, x)
    x = np.matmul(x, W2) + b2
    return x

def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    return PE


    
    
        
    