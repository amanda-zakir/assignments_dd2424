import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pickle
import copy
import warnings
#from torch_gradient_computations import *

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
    mean_X =  np.mean(X, axis=1).reshape(X.shape[0], 1)
    std_X =  np.std(X, axis=1).reshape(X.shape[0], 1)
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
    f = Fs.shape[0]
    n_p = (X.shape[0]//f) * (X.shape[1]//f)
    n = X.shape[3]
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

def InitializeParameters(L, m, rng):
    """
    L: Number of layers
    K: Number of classes
    d: Number of neurons in hidden layer
    m: Number of input features
    """
    net_params = {}
    net_params['Fs'] = [None]
    net_params['W'] = [None] * L
    net_params['b'] = [None] * L
    net_params['Fs'] = rng.normal(0, 1/np.sqrt(3*f*f), (f, f, 3, nf))
    net_params['bf'] = np.zeros((nf, 1))

    for i in range(L):
        if i == 0:
            net_params['W'][i] = np.sqrt(2/m)*rng.standard_normal(size=(m, d)).astype(np.float32)
            net_params['b'][i] = np.zeros((d, 1)).astype(np.float32)
        else:
            net_params['W'][i] = np.sqrt(2/d)*rng.standard_normal(size=(K, d)).astype(np.float32)
            net_params['b'][i] = np.zeros((K, 1)).astype(np.float32)
    return net_params


def ForwardPass(MX, Fs, net_params):
    n_p = MX.shape[0]   
    h_conv = np.zeros((n_p, nf, n), dtype=np.float32)
    nf= Fs.shape[3]
    
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

    return grad_W1, grad_b1, grad_W2, grad_b2, grad_Fs_flat

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

    TrainX = X_all[:, :sample_size]
    TrainY = Y_all[:, :sample_size]
    Trainy = y_all[:sample_size]

    ValX = X_all[:, sample_size:]
    ValY = Y_all[:, sample_size:]
    Valy = y_all[sample_size:]

    TestX, TestY, Testy = LoadBatch('test_batch')
    
    mean_X, std_X = NormalizeData(TrainX)
    TrainX = (TrainX - mean_X) / std_X
    ValX = (ValX - mean_X) / std_X
    TestX = (TestX - mean_X) / std_X

    return TestX, TrainX, TrainY, Trainy, TestY, Testy, ValY, Valy

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
            trained_net['Fs'] -= eta * grads['Fs']

data = LoadBatch('data_batch_1')
X, Y, y = data

net_params= InitializeParameters(2, 10, np.random.default_rng(seed=0))
W = net_params['W']
b = net_params['b']
Fs = net_params['Fs']   

f = Fs.shape[0]
nf = Fs.shape[3]
n = X.shape[1]
X_ims = X.reshape((32, 32, 3, n), order='C')        
n_p = (X_ims.shape[0]//f) * (X_ims.shape[1]//f)
d = n_p * nf
K = Y.shape[0]
L = 2

TestX, TrainX, TrainY, Trainy, TestX, TestY, Testy, ValX, ValY, Valy = DataSplit(X, Y, y, sample_size=1)
MX_TrainX, MX_TestX, MX_ValX = PreComputeMX(TrainX, TestX, ValX, Fs)

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