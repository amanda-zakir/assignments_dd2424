import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pickle
import copy
import warnings
from torch_gradient_computations import *

def LoadBatch(filename):
    # Load a batch of training data
    cifar_dir = 'dataset/cifar-10-batches-py/'

    with open(cifar_dir + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    
    X = dict[b'data'].astype(np.float64) / 255.0
    Y = dict[b'labels']
    
    y = np.array(Y)
    Y = np.eye(np.unique(y).shape[0])[Y].astype(np.float32)

    X = X.reshape(-1, 3, 32, 32).transpose(2, 3, 1, 0)
    
    return X, Y, y

def NormalizeData(X):
    mean_X =  np.mean(X, axis=1)
    std_X =  np.std(X, axis=1)
    return mean_X, std_X

def Convolution(X, Fs):
    f = Fs.shape[0]
    n = X.shape[3]
    nf = Fs.shape[3]
    output = np.zeros((X.shape[0]//f,X.shape[1]//f, nf, n), dtype=np.float32)
    
    for i in range(n):
        for j in range(nf):
            for l in range(X.shape[1]//f):
                for k in range(X.shape[0]//f):
                    output[k,l,j,i] = np.sum(X[k*f:(k+1)*f, l*f:(l+1)*f, :, i] * Fs[:,:,:,j])
    return output

def MXConvolution(X, Fs):
    """
    f = Fs.shape[0]
    n_p = (X.shape[0]//f) * (X.shape[1]//f)
    n = X.shape[3]
    print("n_p:", n_p)
    MX = np.zeros((n_p, f*f*3, n), dtype=np.float32)
    X_patch = np.zeros((f, f, 3), dtype=np.float32)

    for i in range(n):
            count = 0
            for k in range(X.shape[0]//f):
                for l in range(X.shape[1]//f):
                    X_patch = X[k*f: (k+1)*f, l*f:(l+1)*f, :, i]
                    MX[count, :, i] = X_patch.reshape((1, f*f*3), order='C')
                    count += 1
    return MX
    """
    # X shape: (Height, Width, Channels, Num_Images)
    # Ensure X is float32 as requested
    X = X.astype(np.float32)
    
    f = Fs.shape[0]
    h, w, c, n = X.shape
    
    # Calculate n_p based on stride f
    n_p_h = h // f
    n_p_w = w // f
    n_p = n_p_h * n_p_w
    
    # Debug print to catch which dimension is failing
    if n_p != 64:
        print(f"WARNING: Unexpected n_p: {n_p} (Image size: {h}x{w})")
    
    MX = np.zeros((n_p, f * f * c, n), dtype=np.float32)

    for i in range(n):
        count = 0
        for k in range(n_p_h):
            for l in range(n_p_w):
                # Extract the patch
                X_patch = X[k*f : (k+1)*f, l*f : (l+1)*f, :, i]
                # order='C' is standard, just ensure filters match it
                MX[count, :, i] = X_patch.reshape(f * f * c, order='C')
                count += 1
    return MX

def InitializeParameters(L, m, K, d, nf, f, rng):
    """
    L: Number of layers
    K: Number of classes
    d: Number of neurons in hidden layer
    m: Number of hidden neurons
    """
    net_params = {}
    net_params['Fs'] = [None]
    net_params['W'] = [None] * L
    net_params['b'] = [None] * L
    net_params['Fs'] = rng.normal(0, 1/np.sqrt(3*f*f), (f, f, 3, nf))
    #net_params['bf'] = np.zeros((nf, 1))

    for i in range(L):
        if i == 0:
            net_params['W'][i] = (np.sqrt(2/d) * rng.standard_normal(size=(m, d))).astype(np.float32)
            net_params['b'][i] = np.zeros((m, 1)).astype(np.float32)
        else:
            net_params['W'][i] = (np.sqrt(2/m) * rng.standard_normal(size=(K, m))).astype(np.float32)
            net_params['b'][i] = np.zeros((K, 1)).astype(np.float32)
    return net_params


def ForwardPass(MX, Fs, net_params):
    n_p = MX.shape[0]   
    nf = Fs.shape[3]
    n = MX.shape[2]
    h_conv = np.zeros((n_p, nf, n), dtype=np.float32)
    nf= Fs.shape[3]
    Fs_flat = Fs.reshape((f*f*3, nf), order='C')
    
    fp_data = {}
    fp_data['conv_flat'] = [None]
    fp_data['X1'] = [None]
    fp_data['P'] = [None]
    
    for i in range(n):
        h_conv[:,:, i] = np.matmul(MX[:,:, i], Fs_flat) 

    h_conv_relu = np.maximum(0, h_conv)
    h_flat = np.reshape(h_conv_relu, (n_p * nf, n), order='C')
    z_1 = np.matmul(net_params['W'][0], h_flat) + net_params['b'][0]
    x_1 = np.maximum(np.zeros(np.shape(z_1)), z_1)
    s = np.matmul(net_params['W'][1], x_1) + net_params['b'][1]

    P = np.exp(s) / np.sum(np.exp(s), axis=0)
    
    fp_data['conv_flat'] = h_flat
    fp_data['X1'] = x_1
    fp_data['P'] = P
    return fp_data

def ComputeCrossEntropyLoss(P, Y):
    nn = np.shape(P)[1]
    y_idx = Y.astype(int)
    correct_class_probs = P[y_idx, np.arange(nn)]
    loss = np.mean(-np.log(correct_class_probs + 1e-10)) 
    return loss

def BackwardPass(MX, net_params, fp_data, Y, lam):
    n = Y.shape[1]
    P = fp_data['P']
    
    G = (1/n) * (P - Y)

    grad_W2 = G @ fp_data['X1'].T 
    grad_b2 = np.sum(G, axis=1, keepdims=True)
    
    G_hidden = net_params['W'][1].T @ G
    G_relu = G_hidden * (fp_data['X1'] > 0)
    
    grad_W1 = G_relu @ fp_data['conv_flat'].T 
    grad_b1 = np.sum(G_relu, axis=1, keepdims=True)
    
    G_batch = net_params['W'][0].T @ G_relu
    G_batch = G_batch * (fp_data['conv_flat'] > 0)
    GG = G_batch.reshape((n_p, nf, n), order='C')
    
    MXt = np.transpose(MX, (1, 0, 2))
    print("MXt shape:", MXt.shape)
    
    grad_Fs_flat = np.einsum('ijn, jln -> il', MXt, GG, optimize=True) 

    grads = {
    'W': [grad_W1, grad_W2],
    'b': [grad_b1, grad_b2],
    'Fs_flat': grad_Fs_flat
    }
    return grads

def DataSplit(X, Y, y, sample_size):
    X_all = []
    Y_all = []
    y_all = []

    for i in range(1, 6):
        X, Y, y = LoadBatch(f'data_batch_{i}')
        X_all.append(X)
        Y_all.append(Y)
        y_all.append(y)

    X_all = np.concatenate(X_all, axis=3)
    Y_all = np.concatenate(Y_all, axis=1)
    y_all = np.concatenate(y_all, axis=0)

    TrainX = X_all[:, :, :, :sample_size]
    TrainY = Y_all[:, :sample_size]
    Trainy = y_all[:sample_size]

    ValX = X_all[:, :, :, sample_size:]
    ValY = Y_all[:, sample_size:]
    Valy = y_all[sample_size:]

    TestX, TestY, Testy = LoadBatch('test_batch')

    return TestX, TestY, Testy, TrainX, TrainY, Trainy, ValX, ValY, Valy

def PreComputeMX(TrainX, TestX, ValX, Fs):
    MX_TrainX = MXConvolution(TrainX, Fs)
    MX_TestX = MXConvolution(TestX, Fs)
    MX_ValX = MXConvolution(ValX, Fs)
    
    return MX_TrainX, MX_TestX, MX_ValX

def TrainNetwork(MX_Train, TrainY, init_net, net_params):

    trained_net = copy.deepcopy(init_net)

    n_batch = net_params['batch_size']
    n = MX_Train.shape[2]
    eta = net_params['eta']
    epochs = net_params['n_epochs']
    lam = net_params['lam']
    
    for epoch in range(epochs):
        indices = np.random.permutation(MX_Train.shape[2]) 
        
        for j in range(n // n_batch):
            inds = indices[j*n_batch : (j+1)*n_batch]
            
            MX_batch = MX_Train[:, :, inds] 
            Y_batch = TrainY[:, inds]
        
            fp_data = ForwardPass(MX_batch, trained_net) 
            grads = BackwardPass(MX_batch, trained_net, fp_data, Y_batch, net_params['lam'], MX_batch)
            
            for l in range(len(trained_net['W'])):
                trained_net['W'][l] -= eta * grads['W'][l]
                trained_net['b'][l] -= eta * grads['b'][l]
            trained_net['Fs'] -= eta * grads['Fs_flat']

data = LoadBatch('data_batch_1')
X, Y, y = data
f = 4
nf = 3
n = X.shape[3] 
n_p = (X.shape[0]//f) * (X.shape[1]//f)
d = n_p * nf
K = Y.shape[0]
L = 2

net_params= InitializeParameters(L, 10, K, d, nf, f, np.random.default_rng(seed=0))
W = net_params['W']
b = net_params['b']
Fs = net_params['Fs']   




TestX, TestY, Testy, TrainX, TrainY, Trainy, ValX, ValY, Valy = DataSplit(X, Y, y, sample_size=1)
MX_TrainX, MX_TestX, MX_ValX = PreComputeMX(TrainX, TestX, ValX, Fs)

torch_grads = ComputeGradsWithTorch(TrainX, Trainy, MX_TrainX, Fs, net_params)
my_grads = BackwardPass(MX_TrainX, net_params, ForwardPass(MX_TrainX, Fs, net_params), TrainY, lam=0.0)

eps = 1e-10

diff_W1 = np.abs(my_grads['W'][0] - torch_grads['W'][0]) / max(eps, np.sum(np.sum(np.abs(my_grads['W'][0]) + np.abs(torch_grads['W'][0]), axis=0), axis=0))
diff_b1 = np.abs(my_grads['b'][0] - torch_grads['b'][0])/ max(eps, np.sum(np.abs(my_grads['b'][0]) + np.abs(torch_grads['b'][0])))
diff_Fs = np.abs(my_grads['Fs_flat'] - torch_grads['Fs_flat']) / max(eps, np.sum(np.abs(my_grads['Fs_flat']) + np.abs(torch_grads['Fs_flat'])))

diff_W2 = np.abs(my_grads['W'][1] - torch_grads['W'][1]) / max(eps, np.sum(np.sum(np.abs(my_grads['W'][1]) + np.abs(torch_grads['W'][1]), axis=0), axis=0)) 
diff_b2 = np.abs(my_grads['b'][1] - torch_grads['b'][1])/ max(eps, np.sum(np.abs(my_grads['b'][1]) + np.abs(torch_grads['b'][1])))

print("Relative difference in W1: ", diff_W1)
print("Relative difference in b1: ", diff_b1)
print("Relative difference in Fs: ", diff_Fs)
print("Relative difference in W2: ", diff_W2)
print("Relative difference in b2: ", diff_b2)

"""
MX_TrainX, MX_TestX = PreComputeMX(TrainX, TestX)

n_p = MX.shape[0]

Fs_flat = Fs.reshape((f*f*3, nf), order='C')
conv_outputs_mat = np.zeros((n_p, nf, n))
conv_outputs = Convolution(X_ims, Fs)

for i in range(n):
    conv_outputs_mat[:,:, i] = np.matmul(MX[:,:, i], Fs_flat)

conv_outputs_flat = conv_outputs.reshape((n_p, nf, n), order='C')

conv_outputs_mat = np.einsum('ijn, jl ->iln', MX, Fs_flat, optimize=True)
#print(conv_outputs_flat - conv_outputs_mat)

print(load_data.files)
print("W1 shape:", load_data['W1'].shape)
print("b1 shape:", load_data['b1'].shape)
print("W2 shape:", load_data['W2'].shape)
print("b2 shape:", load_data['b2'].shape)

params = InitializeParameters(2, 10, 10, n_p * nf, np.random.default_rng(seed=0))

net_params = {}
net_params['W'] = [load_data['W1'], load_data['W2']]
net_params['b'] = [load_data['b1'], load_data['b2']]

results = ForwardPass(MX, Fs, net_params)

print("Conv output shape:", results['conv_flat'].shape)
print("Debug conv output shape:", load_data['conv_flat'].shape)
print("X1 shape:", results['X1'].shape)
print("Debug X1 shape:", load_data['X1'].shape)
"""
"""
if np.allclose(results['conv_flat'], load_data['conv_flat']):
    print("Step 1 (Convolution) is Correct!")
else:
    print("Step 1 Mismatch. Check your reshape order ('C' vs 'F').")

if np.allclose(results['X1'], load_data['X1']):
    print("Step 1 (Fully Connected Layer) is Correct!")
else:
    print("Step 1 Mismatch. Check your reshape order ('C' vs 'F').")

if np.allclose(results['P'], load_data['P']):
    print("Step 1 (Softmax Layer) is Correct!")
else:
    print("Step 1 Mismatch. Check your reshape order ('C' vs 'F').")

grad_W2, grad_b2, grad_W1, grad_b1, grad_Fs_flat = BackwardPass(net_params, results, load_data['Y'], lam=0.0)
print("Fs grad_Fs_flat.shape:", grad_Fs_flat.shape)

target = load_data['grad_Fs_flat']
print("Target shape:", target.shape)

rel_error = np.abs(grad_Fs_flat - target) / (np.maximum(1e-8, np.abs(grad_Fs_flat) + np.abs(target)))

print(f"Max Relative Error: {np.max(rel_error)}")

if np.max(rel_error) < 1e-7:
    print("✅ Convolution Gradient is CORRECT.")
else:
    print("❌ Mismatch detected. Check your einsum or your 1/n scaling.")
"""