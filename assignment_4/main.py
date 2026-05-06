import numpy as np

#0.1 Read in the data 
book_dir = './'
book_fname = book_dir + 'goblet_book.txt'
fid = open(book_fname, "r")
book_data = fid.read()
fid.close()

unique_chars = list(set(book_data))
K = len(unique_chars)

char_to_ind = {char: i for i, char in enumerate(unique_chars)}
ind_to_char = {i: char for i, char in enumerate(unique_chars)}

#0.2 Set hyper-parameters & initialize RNNs parameters
m = 100
eta = 0.001
seq_length = 25 
b = np.zeros((m,1))
c = np.zeros((K,1))

rng = np.random.default_rng()
Bitgen = type(rng.bit_generator)
seed = 42
rng.bit_generator.state = Bitgen(seed).state

U = (1/np.sqrt(2*K))*rng.standard_normal(size = (m, K))
W = (1/np.sqrt(2*m))*rng.standard_normal(size = (m, m))
V = (1/np.sqrt(m))*rng.standard_normal(size = (K, m)) 

RNN = {"m" : m, "eta" : eta, "seq_length" : seq_length, "b" : b, "c" : c, "U" : U, "W" : W, "V" : V}

#0.3 Synthesize text from your randomly initialized RNN
def SynthesizeText(RNN, h0, x0, n):
    W, U, V  = RNN['W'], RNN['U'], RNN['V']
    b, c = RNN['b'], RNN['c']

    h_prev_t = h0
    x_t = x0
    indices = []

    Y = np.zeros((K, n))

    for t in range(n):
        a_t = W @ h_prev_t + U @ x_t + b 
        h_t = np.tanh(a_t)
        o_t = V @ h_t + c 
        p = np.exp(o_t) / np.sum(np.exp(o_t), axis=0, keepdims=True)

        cp = np.cumsum(p, axis=0)
        a = rng.uniform(size=1)
        ii = np.argmax(cp - a > 0)
        indices.append(ii)

        x_t = np.zeros((K, 1))
        x_t[ii] = 1
    
        Y[:, t] = x_t.flatten()

    return Y 

#0.4 Forward and Backward pass 

h0 = np.zeros((m, 1))
x0 = np.zeros((K, 1))

Y = SynthesizeText(RNN, h0, x0, 200)

indices = np.argmax(Y, axis=0)

print(f"Total unique characters (K): {K}")

text = "".join([str(ind_to_char[char]) for char in indices])

print(text)