import numpy as np
import copy
import pickle
from torch_gradient_computations import *
import time 
import matplotlib.pyplot as plt

"""
debug_file =  "debug_info.npz"
load_data = np.load(debug_file)
X = load_data['X']
Fs = load_data['Fs']
#print(Fs.shape)
Y = load_data['Y']
debug_Fs_grad_flat = load_data['grad_Fs_flat']
#n = X.shape[1]
debug_W1 = load_data['grad_W1']

X_ims = np.transpose(X.reshape((32, 32, 3, n), order='F'), (1, 0, 2, 3))
"""

def LoadBatch(filename):
    cifar_dir = 'cifar-10-batches-py/'

    with open(cifar_dir + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        X = dict[b'data'].astype(np.float64) / 255.0
        Y = dict[b'labels']
        y = np.array(Y)
        Y = np.eye(np.unique(y).shape[0])[Y]
    return X.T, Y.T, y

def NormalizeData(X):
    mean_X =  np.mean(X, axis=1).reshape(X.shape[0], 1)
    std_X =  np.std(X, axis=1).reshape(X.shape[0], 1)
    return mean_X, std_X


def DotProdConv(X_ims, Fs):
    f = Fs.shape[0] # filter size
    n = X_ims.shape[3] # number of images
    nf = Fs.shape[3] # number of filters
    output = np.zeros((X_ims.shape[0]//f,X_ims.shape[1]//f, nf, n))
    
    for i in range(n):
        for j in range(nf):
            for l in range(X_ims.shape[1]//f):
                for k in range(X_ims.shape[0]//f):
                    output[k,l,j,i] = np.sum(X_ims[k*f:(k+1)*f, l*f:(l+1)*f, :, i] * Fs[:,:,:,j])       
    return output


def MXConv(X_ims, Fs):
    f = Fs.shape[0] # filter size
    n = X_ims.shape[3] # number of images
    nf = Fs.shape[3] # number of filters
    n_rows = X_ims.shape[0]//f
    n_cols = X_ims.shape[1]//f
    n_p = n_rows * n_cols

    MX = np.zeros((n_p, f*f*3, n))
    X_patch = np.zeros((f,f,3,n)) #patch of the image

    for i in range(n):
        count = 0
        for k in range(n_rows):
            for l in range(n_cols):
                X_patch = X_ims [k*f: (k+1)*f, l*f:(l+1)*f, :, i]
                MX[count, :, i] = X_patch.reshape((1, f*f*3), order='C')
                count += 1
    print(MX.shape)
    return MX

def ConvMatrixMult(X_ims, MX, Fs):
    Fs_flat = Fs.reshape((f*f*3, nf), order='C')
    n = X_ims.shape[3]
    conv_outputs_mat = np.zeros((n_p, nf, n))
    
    for i in range(n):
        conv_outputs_mat[:, :, i] = np.matmul(MX[:, :, i], Fs_flat)

    #conv_outputs_mat = np.einsum('ijn, jl ->iln', MX, Fs_flat, optimize=True)

    return conv_outputs_mat 

def ForwardPass(X_ims, MX, net):
    L = len(net) + 1
    n = X_ims.shape[3]
    fp_data = {}
    fp_data['input'] = [None] * L
    fp_data ['output'] = [None] * L
    
    conv_outputs = ConvMatrixMult(X_ims, MX, net['Fs'])
    #print("conv_outputs:", conv_outputs.shape)
    conv_outputs_flat = conv_outputs.reshape((n_p*nf, n), order='C')
    fp_data['input'][0] = conv_outputs_flat
    hi = np.fmax(conv_outputs_flat, 0)
    fp_data['output'][0] = hi
    s_1 =  net['W'][0] @ hi + net['b'][0]
    fp_data['input'][1] = s_1
    x1 = np.fmax(s_1, 0)
    fp_data['output'][1] = x1
    s_2 = net['W'][1] @ x1 + net['b'][1]
    fp_data['input'][2] = s_2
    P = np.exp(s_2) / np.sum(np.exp(s_2), axis=0, keepdims=True)
    #print("Anal P:", P[:5,0])
    fp_data['output'][2] = P

    return P, fp_data

def BackwardPass(X_ims, Y_smooth, MX, net, lam):
    n = X_ims.shape[3] # number of images
    P, fp_data = ForwardPass(X_ims, MX, net)

    G = P - Y_smooth

    loss = ComputeCrossEntropyLoss(P, Y_smooth)
    #print("Loss analytical:", loss)

    grad_W2 = (1/n) * G @ fp_data['output'][1].T + (2 * lam * net['W'][1])
    grad_b2 = np.sum(G, axis=1, keepdims=True) * (1/n)

    G = net['W'][1].T @ G
    G = G * (fp_data['output'][1] > 0)

    grad_W1 = (1/n) * G @ fp_data['output'][0].T + (2 * lam * net['W'][0])
    grad_b1 = np.sum(G, axis=1, keepdims=True) * (1/n)

    G_batch = net['W'][0].T @ G
    G_batch = G_batch * (fp_data['output'][0] > 0)

    GG = G_batch.reshape((n_p, nf, n), order='C')
    MXt = np.transpose(MX, (1, 0, 2))

    grad_Fs_flat = (1/n) * np.einsum('ijn, jln ->il', MXt, GG, optimize=True)

    grad_Fs = grad_Fs_flat.reshape((f,f,3,nf), order='C') + (2 * lam * net['Fs'])
    grad_bf = np.sum(GG, axis=(0, 2)).reshape(nf, 1) / n

    grads = {'W': [grad_W1, grad_W2], 'b': [grad_b1, grad_b2], 'Fs': grad_Fs, 'bf' : grad_bf}

    return grads 

def InitializeParameters(L, d, rng):
    net_params = {}
    net_params['Fs'] = [None]
    net_params['W'] = [None] * len(L)
    net_params['b'] = [None] * len(L)
    net_params['Fs'] = rng.normal(0, 1/np.sqrt(3*f*f), (f, f, 3, nf))
    net_params['bf'] = np.zeros((nf, 1))

    for i in range(len(L)):
        if i == 0:
            net_params['W'][i] = rng.normal(0, 1/np.sqrt(d), (L[i],d))
            net_params['b'][i] = np.zeros((L[i],1))
        else:
            net_params['W'][i] = rng.normal(0, 1/np.sqrt(L[i-1]), (L[i], L[i-1]))
            net_params['b'][i] = np.zeros((L[i],1))
    return net_params

def PreComputeMX(TrainX, TestX, ValX, Fs):
    MX_TrainX = MXConv(TrainX, Fs)
    MX_TestX = MXConv(TestX, Fs)
    MX_ValX = MXConv(ValX, Fs)
    
    return MX_TrainX, MX_TestX, MX_ValX

def ComputeCrossEntropyLoss(P, Y_smooth):
    """
    nn = P.shape[1]
    print("anal n:",nn)
    y_idx = np.array(Y).astype(int)
    correct_class_probs = P[y_idx, np.arange(nn)]
    loss = -np.mean(np.log(correct_class_probs))
    return loss

    """
    return - np.sum(Y_smooth * np.log(P)) / Y_smooth.shape[1]

def ComputeAccuracy(P, y):
    acc = 0
    predicted_classes = np.argmax(P, axis=0)
    real_classes = y
    acc = np.mean(predicted_classes == real_classes) * 100
    return str(acc) + " %"

def CyclicLearningRate(eta_min, eta_max, t, initial_n_s):
    # Implementation for cyclic learning rate
    
    current_n_s = initial_n_s
    t_remaining = t
    while t_remaining >= 2 * current_n_s:
        t_remaining -= 2 * current_n_s
        current_n_s *= 2

    l = t // (2 * current_n_s)

    if t_remaining < current_n_s:
        eta_t = eta_min + (t_remaining ) / current_n_s * (eta_max - eta_min)
    else:
        eta_t = eta_max - ((t_remaining - current_n_s)) / current_n_s * (eta_max - eta_min)
    return eta_t

def LabelSmoothing(Y, K, epsilon):

    N = Y.shape[1]
    
    off_value = epsilon / (K - 1)
    Y_smooth = np.full((K, N), off_value)
    
    target_inds = np.argmax(Y, axis=0)

    Y_smooth[target_inds, np.arange(N)] = 1 - epsilon
    
    return Y_smooth

def MiniBatchGD(X, Y, Y_smooth, y, ValX, ValY, Valy, TrainMX, ValMX, GDperms, init_net, lam, rng):
    trained_net = copy.deepcopy(init_net)
    n = X.shape[3]
    n_batch, n_epochs, eta_min, eta_max, n_s = GDperms['n_batch'], GDperms['n_epochs'], GDperms['eta_min'], GDperms['eta_max'], GDperms['n_s']

    t = 0

    train_loss = []
    test_loss = []
    acc_train = []
    acc_test = []
    update_steps = []

    for epoch in range(n_epochs):
        indices = rng.permutation(n)  
        X_rand = X[:,:,:,indices]
        Y_rand = Y[:,indices]
        Y_smooth_rand = Y_smooth[:, indices]
        TrainMX_rand = TrainMX[:,:,indices]
        for j in range(n // n_batch):
            t += 1
            eta = CyclicLearningRate(eta_min, eta_max, t, n_s)
            #print(eta)
            start = j * n_batch
            end = (j + 1) * n_batch
            
            X_batch = X_rand[:, :, :, start:end]
            Y_batch = Y_rand[:, start:end]
            Y_smooth_batch = Y_smooth[:,start:end]
            
            MX_batch = TrainMX_rand[:, :, start:end]
            
            # Forward & Backward
            grads = BackwardPass(X_batch, Y_batch, MX_batch, trained_net, lam)
            
            # Update weights
            for i in range(len(trained_net['W'])):
                trained_net['W'][i] -= eta * grads['W'][i]
                trained_net['b'][i] -= eta * grads['b'][i]
            trained_net['Fs'] -= eta * grads['Fs']
            trained_net['bf'] -= eta * grads['bf']
        
            if t % 500 == 0 or t==1:
                P_train, _ = ForwardPass(X, TrainMX, trained_net)
                P_test, _ = ForwardPass(ValX, ValMX, trained_net)
                
                tr_loss = ComputeCrossEntropyLoss(P_train, Y)
                ts_loss = ComputeCrossEntropyLoss(P_test, ValY)
                
                tr_acc = float(ComputeAccuracy(P_train, y).split()[0]) / 100
                ts_acc = float(ComputeAccuracy(P_test, Valy).split()[0]) / 100

                train_loss.append(tr_loss)
                test_loss.append(ts_loss)
                acc_train.append(tr_acc)
                acc_test.append(ts_acc)
                update_steps.append(t) 
    print(t)   
    return trained_net, np.array(train_loss), np.array(test_loss), np.array(acc_train), np.array(acc_test), np.array(update_steps)

"""
f = Fs.shape[0] # filter size
#n = X_ims.shape[1] # number of images
nf = Fs.shape[3] # number of filters
n_rows = X_ims.shape[0]//f
n_cols = X_ims.shape[1]//f
n_p = n_rows * n_cols

Fs_flat = Fs.reshape((f*f*3, nf), order='C')
conv_outputs_mat = np.zeros((n_p, nf, n))
#MX = MXConv(X_ims, Fs)
ground_truth_conv_outputs = DotProdConv(X_ims, Fs)
conv_outputs = load_data['conv_outputs']
#conv_outputs = ConvMatrixMult(X_ims, MX, Fs)

#Forward pass debug
debug_conv_flat = load_data['conv_flat']
debug_X1 = load_data['X1']
debug_P = load_data['P']
W1 = load_data['W1']
#print("W1 shape: ", W1.shape)
b1 = load_data['b1']
W2 = load_data['W2']
b2 = load_data['b2']
    


#print("Difference between debug outputs and forward pass outputs:", (debug_P - P))
#print("Debug conv_flat: ", fp_data['output'][0] - debug_conv_flat)
#print("Debug X1: ", fp_data['output'][1] - debug_X1)
#print("Debug P: ", fp_data['output'][2] - debug_P)

#Backward pass debug
#grads = BackwardPass(X_ims, Y, MX, {'W' : [W1,W2], 'b' : [b1, b2], 'Fs' : Fs}, lam=0.0)

#print("Difference between debug fs flat and analytical: ", np.linalg.norm(debug_Fs_grad_flat - grads['Fs'].reshape(debug_Fs_grad_flat.shape, order='C')))
#print("Difference between debug fs flat and analytical: ", np.linalg.norm(debug_W1 - grads['W'][0]))

#Comparing gradients
"""
"""
nh = 50
L = [nh, 10]
n_rows = X_ims.shape[0]//f
n_cols = X_ims.shape[1]//f
n_p = n_rows * n_cols
d = n_p * nf
lam = 0.0

test_params = InitializeParameters(L, d, rng)

grads = BackwardPass(X_ims, Y, MX, test_params, lam=0.0)

my_grads = grads
torch_grads = ComputeGradsWithTorch(X_ims, Y, MX, test_params, lam)

eps = 1e-10

diff_W1 = np.sum(np.abs(my_grads['W'][0] - torch_grads['W'][0]))
diff_b1 = np.abs(my_grads['b'][0] - torch_grads['b'][0])/ max(eps, np.sum(np.abs(my_grads['b'][0]) + np.abs(torch_grads['b'][0])))
diff_Fs = np.abs(my_grads['Fs'] - torch_grads['Fs']) / max(eps, np.sum(np.abs(my_grads['Fs']) + np.abs(torch_grads['Fs'])))

diff_W2 = np.abs(my_grads['W'][1] - torch_grads['W'][1]) / max(eps, np.sum(np.sum(np.abs(my_grads['W'][1]) + np.abs(torch_grads['W'][1]), axis=0), axis=0)) 
diff_b2 = np.abs(my_grads['b'][1] - torch_grads['b'][1])/ max(eps, np.sum(np.abs(my_grads['b'][1]) + np.abs(torch_grads['b'][1])))

"""
"""
print("Relative difference in W1: ", diff_W1)
print("Relative difference in b1: ", np.max(diff_b1))
print("Relative difference in Fs: ", np.max(diff_Fs))
print("Relative difference in W2: ", np.max(diff_W2))
print("Relative difference in b2: ", np.max(diff_b2))
"""
"""
"""

#print("Analytical conv:", fp_data['input'][0][:5,0])

#print("Analytical patch:", MX[0, :, 0][:10])
#print("Anal Fs shape:", test_params['Fs'].shape)
#print("Anal Fs shape:", test_params['Fs'][0,0,0,:])

#Exercise 3
TrainX1, TrainY1, Trainy1 = LoadBatch('data_batch_1')
TrainX2, TrainY2, Trainy2 = LoadBatch('data_batch_2')
TrainX3, TrainY3, Trainy3 = LoadBatch('data_batch_3')
TrainX4, TrainY4, Trainy4 = LoadBatch('data_batch_4')
TrainX5, TrainY5, Trainy5 = LoadBatch('data_batch_5')

TestX, TestY, Testy = LoadBatch('test_batch')

TrainX_all = np.concatenate((TrainX1, TrainX2, TrainX3, TrainX4, TrainX5), axis=1)
TrainY_all = np.concatenate((TrainY1, TrainY2, TrainY3, TrainY4, TrainY5), axis=1)
Trainy_all = np.concatenate((Trainy1, Trainy2, Trainy3, Trainy4, Trainy5), axis=0)

# Split into 49,000 training and 1,000 validation (or keep 5k if preferred)
TrainX = TrainX_all[:, :49000]
TrainY = TrainY_all[:, :49000]
Trainy = Trainy_all[:49000]

ValX = TrainX_all[:, 49000:]
ValY = TrainY_all[:, 49000:]
Valy = Trainy_all[49000:]

#Preprocess data

K = TrainY_all.shape[0]

rng = np.random.default_rng()
Bitgen = type(rng.bit_generator)
seed = 42
rng.bit_generator.state = Bitgen(seed).state

# Test set should be loaded separately
TestX, TestY, Testy = LoadBatch('test_batch')

mean_X, std_X = NormalizeData(TrainX)
TrainX = (TrainX - mean_X) / std_X
ValX = (ValX - mean_X) / std_X
TestX = (TestX - mean_X) / std_X

n = TrainX.shape[1]

TrainX_ims = TrainX.reshape((32,32,3,n), order='F')
ValX_ims = ValX.reshape((32,32,3,ValX.shape[1]), order='F')
TestX_ims = TestX.reshape((32,32,3,TestX.shape[1]), order='F')

#print(TrainX_ims.shape)



f = 4
nf = 40
n_p = int(32/f)**2
d = n_p * nf
nh = 300
L = [nh, K]
lam = 0.001
n_cycles = 4
n_batch = 100
eta_min = 1e-5
eta_max = 1e-1
epsilon = 0.1

TrainY_Smooth = LabelSmoothing(TrainY, K, epsilon)

net_params = InitializeParameters(L, d, rng)

TrainMX, TestMX, ValMX = PreComputeMX(TrainX_ims, TestX_ims, ValX_ims, net_params['Fs'])

n_s = 800 
n_epochs = 49
print(n_epochs)
GDparams = {
    "n_batch": n_batch, 
    "eta_min": eta_min, 
    "eta_max": eta_max, 
    "n_s": n_s, 
    "n_epochs": n_epochs
}

start_time = time.time()
trained_net, train_loss, test_loss, acc_train, acc_test, steps = MiniBatchGD(TrainX_ims, TrainY, TrainY_Smooth, Trainy, ValX_ims, ValY, Valy, TrainMX, ValMX, GDparams, net_params, lam, rng)
end_time = time.time()
print("Training time: ", round(end_time - start_time, 2), " seconds")
P_test, _ = ForwardPass(TestX_ims, TestMX, trained_net)
print("Test Accuracy: ", ComputeAccuracy(P_test, Testy))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
print("Train loss:", train_loss)
ax1.plot(steps, train_loss, 'g-', label='Training')
ax1.plot(steps, test_loss, 'r-', label='Test')
ax1.set_ylim(0, max(max(train_loss), max(test_loss)) * 1.1)
ax1.set_title('Loss Function')
ax1.set_xlabel('Update step')
ax1.set_ylabel('Loss')
ax1.legend()


ax2.plot(steps,acc_train, 'g-', label='Training')
ax2.plot(steps, acc_test, 'r-', label='Validation')
ax2.set_ylim(0, max(max(acc_train), max(acc_test)) * 1.1)
ax2.set_title('Accuracy')
ax2.set_xlabel('Update step')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.tight_layout()
plt.show()