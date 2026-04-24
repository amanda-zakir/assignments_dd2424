import numpy as np

debug_file = 'debug_info.npz'
load_data = np.load(debug_file)
X = load_data['X']
Fs = load_data['Fs']
debug = load_data['conv_outputs']

f = Fs.shape[0]
nf = Fs.shape[3]
#print("nf:", nf)
n = X.shape[1]

#print(Fs.shape[0])
#print(X.shape)

X_ims = np.transpose(X.reshape((32, 32, 3, n), order='F'), (1, 0, 2, 3))
n_p = (X_ims.shape[0]//f) * (X_ims.shape[1]//f)
#print("n_p:", n_p)

def Convolution(X, Fs):
    n = X.shape[3]
    nf = Fs.shape[3]
    output = np.zeros((X.shape[0]//f,X.shape[1]//f, nf, n))
    
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
    MX = np.zeros((n_p, f*f*3, n))
    X_patch = np.zeros((f, f, 3))

    for i in range(n):
            count = 0
            for k in range(X.shape[0]//f):
                for l in range(X.shape[1]//f):
                    X_patch = X[k*f: (k+1)*f, l*f:(l+1)*f, :, i]
                    MX[count, :, i] = X_patch.reshape((1, f*f*3), order='C')
                    count += 1
    return MX
    """
    f = Fs.shape[0]
    n_p = (X.shape[0]//f) * (X.shape[1]//f)  # Number of patches (64)
    n = X.shape[3]                           # Batch size (5)
    pixels = f*f*3                           # Total pixels per patch (48)
    
    # 1. Change the initialization shape
    MX = np.zeros((n_p, pixels, n))          # (64, 48, 5)
    
    for i in range(n):
        count = 0
        for k in range(X.shape[0]//f):
            for l in range(X.shape[1]//f):
                # Extract the 4x4x3 patch
                X_patch = X[k*f:(k+1)*f, l*f:(l+1)*f, :, i]
                
                # 2. Store the patch in the first dimension (count)
                # Ensure we reshape to (48,) so it fits into MX[count, :, i]
                MX[count, :, i] = X_patch.reshape(pixels, order='C')
                
                count += 1
    return MX

def InitializeParameters(L, K, d, m, rng):
    net_params = {}
    net_params['W'] = [None] * L
    net_params['b'] = [None] * L

    for i in range(L):
        if i == 0:
            net_params['W'][i] = (1/np.sqrt(d))*rng.standard_normal(size=(d, m))
            net_params['b'][i] = np.zeros((d, 1))
        else:
            net_params['W'][i] = (1/np.sqrt(m))*rng.standard_normal(size=(d,m))
            net_params['b'][i] = np.zeros((K, 1))
    print(np.shape(net_params))
    return net_params


def ForwardPass(MX, Fs, net_params):
    n_p = MX.shape[0]   
    Fs_flat = Fs.reshape((f*f*3, nf), order='C')
    h_conv = np.zeros((n_p, nf, n))
    
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

def BackwardPass(net_params, fp_data, Y, lam):
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
    GG = G_batch.reshape((n_p, nf, n), order='C')
    
    MXt = np.transpose(MX, (1, 0, 2))
    print("MXt shape:", MXt.shape)
    
    grad_Fs_flat = np.einsum('ijn, jln -> il', MXt, GG, optimize=True) 

    print("Analytical Sum:", np.sum(np.abs(grad_Fs_flat)))
    print("Target Sum:    ", np.sum(np.abs(load_data['grad_Fs_flat'])))

    return grad_W1, grad_b1, grad_W2, grad_b2, grad_Fs_flat

MX = MXConvolution(X_ims, Fs)
n_p = MX.shape[0]

Fs_flat = Fs.reshape((f*f*3, nf), order='C')
conv_outputs_mat = np.zeros((n_p, nf, n))
conv_outputs = Convolution(X_ims, Fs)

for i in range(n):
    conv_outputs_mat[:,:, i] = np.matmul(MX[:,:, i], Fs_flat)

conv_outputs_flat = conv_outputs.reshape((n_p, nf, n), order='C')

conv_outputs_mat = np.einsum('ijn, jl ->iln', MX, Fs_flat, optimize=True)
#print(conv_outputs_flat - conv_outputs_mat)
"""
print(load_data.files)
print("W1 shape:", load_data['W1'].shape)
print("b1 shape:", load_data['b1'].shape)
print("W2 shape:", load_data['W2'].shape)
print("b2 shape:", load_data['b2'].shape)
"""

params = InitializeParameters(2, 10, 10, n_p * nf, np.random.default_rng(seed=0))

net_params = {}
net_params['W'] = [load_data['W1'], load_data['W2']]
net_params['b'] = [load_data['b1'], load_data['b2']]

results = ForwardPass(MX, Fs, net_params)
"""
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
"""
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

print("First 5 Analytical:", grad_Fs_flat.flatten()[:5])
print("First 5 Target:    ", load_data['grad_Fs_flat'].flatten()[:5])