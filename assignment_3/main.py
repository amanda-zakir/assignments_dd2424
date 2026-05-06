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

    
    return X.T, Y.T, y

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
    f = Fs.shape[0]
    n_p = (X.shape[2]//f) * (X.shape[3]//f)
    n = X.shape[0]
    print(n)
    print("n_p:", n_p)
    MX = np.zeros((n_p, f*f*3, n), dtype=np.float32)
    X_patch = np.zeros((f, f, 3), dtype=np.float32)

    for i in range(n):
            count = 0
            for k in range(X.shape[2]//f):
                for l in range(X.shape[3]//f):
                    X_patch = X[i, :, k*f: (k+1)*f, l*f:(l+1)*f]
                    MX[count, :, i] = X_patch.reshape((1, f*f*3), order='C')
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
    net_params['bf'] = np.zeros((nf, 1))

    for i in range(L):
        if i == 0:
            net_params['W'][i] = (np.sqrt(2/d) * rng.standard_normal(size=(m, d))).astype(np.float32)
            net_params['b'][i] = np.zeros((m, 1)).astype(np.float32)
        else:
            net_params['W'][i] = (np.sqrt(2/m) * rng.standard_normal(size=(K, m))).astype(np.float32)
            net_params['b'][i] = np.zeros((K, 1)).astype(np.float32)
    return net_params


def ForwardPass(MX, Fs, net_params):
    print("MX shape:", MX.shape)
    n_p = MX.shape[0]          # number of patches
    n = MX.shape[2]            # batch size
    nf = Fs.shape[3]           # number of filters
    f = Fs.shape[0]
    c = Fs.shape[2]

    # ---- Flatten filters ----
    Fs_flat = Fs.transpose(2, 0, 1, 3).reshape((c * f * f, nf), order='C')

    # ---- Convolution ----
    h_conv = np.zeros((n_p, nf, n), dtype=np.float32)
    for i in range(n):
        h_conv[:, :, i] = MX[:, :, i] @ Fs_flat

    # ---- ReLU ----
    h_conv_relu = np.maximum(0, h_conv)

    # ---- Flatten: (n_p, nf, n) → (nf, n_p, n) → (nf*n_p, n) ----
    h_conv_perm = np.transpose(h_conv_relu, (1, 0, 2))
    conv_flat = h_conv_perm.reshape(nf * n_p, n, order='C')

    # ---- First FC ----
    z1 = net_params['W'][0] @ conv_flat + net_params['b'][0]
    X1 = np.maximum(0, z1)

    # ---- Second FC ----
    s = net_params['W'][1] @ X1 + net_params['b'][1]

    # ---- Softmax ----
    s_max = np.max(s, axis=0, keepdims=True)
    exp_s = np.exp(s - s_max)
    P = exp_s / np.sum(exp_s, axis=0, keepdims=True)

    # ---- Only required outputs ----
    load_data = {}
    load_data['conv_flat'] = conv_flat   # (n_p * nf, n)
    load_data['X1'] = X1                 # (nh, n)
    load_data['P'] = P                   # (10, n)

    return load_data

def ComputeCrossEntropyLoss(P, Y):
    nn = P.shape[1]
    print(nn)
    y_idx = np.array(Y).astype(int)
    correct_class_probs = P[y_idx, np.arange(nn)]
    loss = -np.mean(np.log(correct_class_probs))
    return loss

def ComputeCost(P, y, network, lam):
    loss = ComputeCrossEntropyLoss(P, y)
    W = network['W']
    reg_term = lam * (np.sum(W[0]**2) + np.sum(W[1]**2))
    return loss + reg_term

def ComputeAccuracy(P, y):
    acc = 0
    predicted_classes = np.argmax(P, axis=0)
    real_classes = y
    acc = np.mean(predicted_classes == real_classes) * 100
    return str(acc) + " %"

def BackwardPass(MX, Y, net_params, fp_data, lam):
    """
    n = Y.shape[1]
    n_p = MX.shape[0]
    nf = net_params['Fs'].shape[3]
    P = fp_data['P'].copy()
    print("MX shape:", MX.shape)
    
    G = (P - Y) / n 

    grad_W2 = (G @ fp_data['X1'].T) + (2 * lam * net_params['W'][1])
    grad_b2 = np.sum(G, axis=1, keepdims=True) 
    
    G_hidden = net_params['W'][1].T @ G
    G_relu = G_hidden * (fp_data['X1'] > 0)
    
    grad_W1 = (G_relu @ fp_data['conv_flat'].T) + (2 * lam * net_params['W'][0])
    grad_b1 = np.sum(G_relu, axis=1, keepdims=True) 
    
    G_batch = net_params['W'][0].T @ G_relu
    G_batch = G_batch * (fp_data['conv_flat'].reshape(n_p * nf, n, order='C') > 0)

    n_p, _, _ = MX.shape 
    nf = net_params['Fs'].shape[3] 
    print("conv_flat shape:", fp_data['conv_flat'].shape)
    print("expected:", n_p * nf, n) 
    GG = G_batch.reshape((nf, n_p, n), order='C') 

    GG = GG.transpose(1, 0, 2) 
    print("GG shape:", GG.shape)
    
    MXt = np.transpose(MX, (1, 0, 2))
    #print("MXt shape:", MXt.shape)
    
    grad_Fs_flat =  np.einsum('ijn, jln -> il', MXt, GG) 
    
    grads = {
        'W': [grad_W1, grad_W2],
        'b': [grad_b1, grad_b2],
        'Fs_flat': grad_Fs_flat
    }

    print("Fs_flat shape:", grad_Fs_flat.shape)

    loss = ComputeCrossEntropyLoss(P, Y)
    print("Loss analytical:", loss)

    return grads
    """
    n = Y.shape[1]
    n_p = MX.shape[0]
    nf = net_params['Fs'].shape[3]

    P = fp_data['P']
    X1 = fp_data['X1']
    conv_flat = fp_data['conv_flat']

    print("Shape of P:", P.shape)
    print("Shape of Y:", Y.shape)

    # ---- Output layer ----
    G = (P - Y) / n

    grad_W2 = G @ X1.T + 2 * lam * net_params['W'][1]
    grad_b2 = np.sum(G, axis=1, keepdims=True)

    # ---- Backprop to hidden layer ----
    G_hidden = net_params['W'][1].T @ G
    G_hidden *= (X1 > 0)   # ReLU derivative

    grad_W1 = G_hidden @ conv_flat.T + 2 * lam * net_params['W'][0]
    grad_b1 = np.sum(G_hidden, axis=1, keepdims=True)

    # ---- Backprop to conv_flat ----
    G_batch = net_params['W'][0].T @ G_hidden

    # ReLU derivative for conv layer
    G_batch *= (conv_flat > 0)

    # ---- Reshape back to (n_p, nf, n) ----
    # Forward did: (n_p, nf, n) → (nf, n_p, n) → (nf*n_p, n)
    # So reverse it:

    GG = G_batch.reshape((nf, n_p, n), order='C')
    GG = np.transpose(GG, (1, 0, 2))   # (n_p, nf, n)

    # ---- Compute gradient for filters ----
    # MX: (n_p, d, n)
    # GG: (n_p, nf, n)

    grad_Fs_flat = np.zeros((MX.shape[1], nf))

    for i in range(n):
        grad_Fs_flat += MX[:, :, i].T @ GG[:, :, i]
    grad_Fs_flat /= n

    # ---- Package gradients ----
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

    X_all = np.concatenate(X_all, axis=0)
    Y_all = np.concatenate(Y_all, axis=1)
    y_all = np.concatenate(y_all, axis=0)

    TrainX = X_all[:sample_size, :, :, :]
    TrainY = Y_all[:, :sample_size]
    Trainy = y_all[:sample_size]

    ValX = X_all[sample_size:, :, :, :]
    ValY = Y_all[:, sample_size:]
    Valy = y_all[sample_size:]

    TestX, TestY, Testy = LoadBatch('test_batch')

    return TestX, TestY, Testy, TrainX, TrainY, Trainy, ValX, ValY, Valy

def PreComputeMX(TrainX, TestX, ValX, Fs):
    MX_TrainX = MXConvolution(TrainX, Fs)
    MX_TestX = MXConvolution(TestX, Fs)
    MX_ValX = MXConvolution(ValX, Fs)
    
    return MX_TrainX, MX_TestX, MX_ValX

def CyclicLearningRate(eta_min, eta_max, t, initial_n_s):
    # Implementation for cyclic learning rate

    current_n_s = initial_n_s
    t_remaining = t
    while t_remaining >= 2 * current_n_s:
        t_remaining -= 2 * current_n_s
        current_n_s *= 2
    
    l = t // (2 * current_n_s)

    if t <= (2*l +1) * current_n_s and t >= 2 * l * current_n_s:
        eta_t = eta_min + (t - 2 * l * current_n_s ) / current_n_s * (eta_max - eta_min)
    else:
        eta_t = eta_max - (t - (2 * l + 1) * current_n_s ) / current_n_s * (eta_max - eta_min)
    return eta_t

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
            trained_net['bf'] -= eta * grads['bf_flat']

def MiniBatchGD(X, Y, y_labels, ValX, ValY, val_y_labels, GDperms, init_net, lam, rng):
    trained_net = copy.deepcopy(init_net)
    n = X.shape[1]
    n_batch, n_epochs, eta_min, eta_max, n_s = GDperms['n_batch'], GDperms['n_epochs'], GDperms['eta_min'], GDperms['eta_max'], GDperms['n_s']

    t = 0

    train_loss = []
    test_loss = []
    acc_train = []
    acc_test = []
    update_steps = []

    for epoch in range(n_epochs):
        indices = rng.permutation(n)  
        for j in range(n // n_batch):
            t += 1
            eta = CyclicLearningRate(eta_min, eta_max, t, n_s)
            j_start, j_end = j * n_batch, (j + 1) * n_batch
            inds = indices[j_start:j_end]
            X_batch, Y_batch = X[:, inds], Y[:, inds]
            
            # Forward & Backward
            P_batch, fp_batch = ForwardPass(X_batch, trained_net)
            grads = BackwardPass(X_batch, Y_batch, P_batch, fp_batch, trained_net, lam)
            
            # Update weights
            for i in range(len(trained_net['W'])):
                trained_net['W'][i] -= eta * grads['W'][i]
                trained_net['b'][i] -= eta * grads['b'][i]
            trained_net['Fs'] -= eta * grads['Fs_flat']
            trained_net['bf'] -= eta * grads['bf_flat']
    
        P_train, _ = ForwardPass(X, trained_net['Fs'], net_params)
        P_test, _ = ForwardPass(ValX, trained_net['Fs'], net_params)
        
        tr_loss = ComputeCrossEntropyLoss(P_train, y_labels)
        ts_loss = ComputeCrossEntropyLoss(P_test, val_y_labels)
        
        tr_acc = float(ComputeAccuracy(P_train, y_labels).split()[0]) / 100
        ts_acc = float(ComputeAccuracy(P_test, val_y_labels).split()[0]) / 100

        train_loss.append(tr_loss)
        test_loss.append(ts_loss)
        acc_train.append(tr_acc)
        acc_test.append(ts_acc)
        update_steps.append(t) 
    
    return trained_net, train_loss, test_loss, acc_train, acc_test, update_steps

data = LoadBatch('data_batch_1')
X, Y, y = data

f = 4
nf = 2
nh = 50
n = X.shape[0] 
n_p = (X.shape[2]//f) * (X.shape[3]//f)
d = n_p * nf
K = Y.shape[0]
L = 2
m = nh

rng = np.random.default_rng()
BitGen = type(rng.bit_generator)
seed = 42
rng.bit_generator.state = BitGen(seed).state

net_params = InitializeParameters(L, m, K, d, nf, f, rng)
W = net_params['W']
b = net_params['b']
Fs = net_params['Fs']   
print("Fs shape:", Fs.shape)

TestX, TestY, Testy, TrainX, TrainY, Trainy, ValX, ValY, Valy = DataSplit(X, Y, y, sample_size=10)

MX_TrainX, MX_TestX, MX_ValX = PreComputeMX(TrainX, TestX, ValX, Fs)
fp_data = ForwardPass(MX_TrainX, Fs, net_params)
my_grads = BackwardPass(MX_TrainX, TrainY, net_params, fp_data, lam=0.0)
torch_grads = ComputeGradsWithTorch(TrainX, Trainy, MX_TrainX, Fs, net_params, fp_data['conv_flat'])

eps = 1e-10
diff_Fs = np.abs(my_grads['Fs_flat'] - torch_grads['Fs_flat'])
#print("Difference in Fs gradients (first 5 values): ", diff_Fs[:5])
#print(net_params['Fs'].shape[3])

diff_W1 = np.abs(my_grads['W'][0] - torch_grads['W'][0]) / max(eps, np.sum(np.sum(np.abs(my_grads['W'][0]) + np.abs(torch_grads['W'][0]), axis=0), axis=0))
diff_b1 = np.abs(my_grads['b'][0] - torch_grads['b'][0])/ max(eps, np.sum(np.abs(my_grads['b'][0]) + np.abs(torch_grads['b'][0])))
diff_Fs = np.abs(my_grads['Fs_flat'] - torch_grads['Fs_flat']) / max(eps, np.sum(np.abs(my_grads['Fs_flat']) + np.abs(torch_grads['Fs_flat'])))

diff_W2 = np.abs(my_grads['W'][1] - torch_grads['W'][1]) / max(eps, np.sum(np.sum(np.abs(my_grads['W'][1]) + np.abs(torch_grads['W'][1]), axis=0), axis=0)) 
diff_b2 = np.abs(my_grads['b'][1] - torch_grads['b'][1])/ max(eps, np.sum(np.abs(my_grads['b'][1]) + np.abs(torch_grads['b'][1])))

print("Relative difference in W1: ", np.max(diff_W1))
print("Relative difference in b1: ", np.max(diff_b1))
print("Relative difference in Fs: ", np.max(diff_Fs))
print("Relative difference in W2: ", np.max(diff_W2))
print("Relative difference in b2: ", np.max(diff_b2))
'''
lam = 0.003
eta_min = 1e-5
eta_max = 1e-1
n_batch = 100
n_cycles = 3 
step = 800 
n = TrainX.shape[1]
n_s = 4 * np.floor(n / n_batch)
GDperms = {'n_batch': n_batch, 'n_epochs': 24, 'eta_min' : eta_min, 'eta_max': eta_max, 'n_s': n_s}




init_net = copy.deepcopy(net_params)

final_net, train_loss, test_loss, acc_train, acc_test, steps = MiniBatchGD(TrainX, TrainY, Trainy, ValX, ValY, Valy, GDperms, init_net, lam, rng)    

P_test, _ = ForwardPass(TestX, final_net, net_params)
test_acc = ComputeAccuracy(P_test, Testy)
print(f"--- FINAL TEST ACCURACY: {test_acc} ---")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(steps, train_loss, 'g-', label='Training loss')
ax1.plot(steps, test_loss, 'r-', label='Testing loss')
ax1.set_ylim(0, max(max(train_loss), max(test_loss)) * 1.1)
ax1.set_title('Cost Function')
ax1.set_xlabel('Update step')
ax1.set_ylabel('Cost')
ax1.legend()

ax2.plot(steps,acc_train, 'g-', label='Training accuracy')
ax2.plot(steps, acc_test, 'r-', label='Testing accuracy')
ax2.set_ylim(0, max(max(acc_train), max(acc_test)) * 1.1)
ax2.set_title('Accuracy')
ax2.set_xlabel('Update step')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.tight_layout()
plt.show()



'''
"""
print("My Grads first 5 Fs values: ", my_grads['Fs_flat'][:5])
print("Torch Grads first 5 Fs values: ", torch_grads['Fs_flat'][:5])



"""

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