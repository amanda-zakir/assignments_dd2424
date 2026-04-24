import json
import matplotlib.pyplot as plt
import numpy as np
import pickle
import copy
import warnings
from torch_gradient_computations import *

def LoadBatch(filename):
    # Load a batch of training data
    cifar_dir = 'datasets/cifar-10-batches-py/'

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

def LoadImage(X):
    X = X.transpose()
    nn = X.shape[1]
    # Reshape each image from a column vector to a 3d array
    X_im = X.reshape((32, 32, 3, nn), order='F')
    X_im = np.transpose(X_im, (1, 0, 2, 3))
    
    # Display the first 5 images
    ni = 5
    fig, axs = plt.subplots(1, 5, figsize=(10, 5))
    for i in range(ni):
        axs[i].imshow(X_im[:, :, :, i])
        axs[i].axis('off')
    plt.pause(1)

def ApplyNetwork(X, network):
    W = network['W']
    b = network['b']
    
    fp_data = {}
    fp_data['hidden'] = [None]
    fp_data['output'] = [None]


    s_1 = np.matmul(W[0], X) + b[0]
    h = np.maximum(np.zeros(np.shape(s_1)), s_1)
    fp_data['hidden'] = s_1
    s = np.matmul(W[1], h) + b[1]
    #print(np.shape(s))
   
    #P = np.array(np.exp(s) / np.sum(np.exp(s)))

    #s_stable = s - np.max(s, axis=0, keepdims=True)
    fp_data['output'] = s
    P = np.exp(s) / np.sum(np.exp(s), axis=0)

    return np.array(P), fp_data

def ComputeLoss(P, y):
    nn = np.shape(P)[1]
    y_idx = y.astype(int)
    correct_class_probs = P[y_idx, np.arange(nn)]
    loss = np.mean(-np.log(correct_class_probs + 1e-10)) 
    return loss

def ComputeCost(P, y, network, lam):
    loss = ComputeLoss(P, y)
    W = network['W']
    reg_term = lam * (np.sum(W[0]**2) + np.sum(W[1]**2))
    return loss + reg_term
    

def ComputeAccuracy(P, y):
    acc = 0
    predicted_classes = np.argmax(P, axis=0)
    real_classes = y
    acc = np.mean(predicted_classes == real_classes) * 100
    return str(acc) + " %"

def BackwardPass(X, Y, P, fp_data, network, lam):
    W = network['W']
    n = X.shape[1]
    #print(P.shape)
    #print(Y.shape)
    G = P - Y

    h = np.maximum(np.zeros(np.shape(fp_data['hidden'])), fp_data['hidden'])
    grad_W_l = (1/n) * G @ h.T + 2 * lam * W[1]
    grad_b_l = (1/n) * np.sum(G, axis=1, keepdims=True)

    G = W[1].T @ G
    G = G * (fp_data['hidden'] > 0)
    
    grad_W_1 = (1/n) * G @ X.T + 2 * lam * W[0]
    grad_b_1 = (1/n) * np.sum(G, axis=1, keepdims=True)

    #Backward pass
    grads = {'W': [grad_W_1, grad_W_l], 'b': [grad_b_1, grad_b_l]}

    return grads

def MiniBatchGD(X, Y, y_labels, ValX, ValY, val_y_labels, GDperms, init_net, lam, rng):
    trained_net = copy.deepcopy(init_net)
    n = X.shape[1]
    n_batch, n_epochs, eta_min, eta_max, n_s = GDperms['n_batch'], GDperms['n_epochs'], GDperms['eta_min'], GDperms['eta_max'], GDperms['n_s']

    t = 0

    train_loss = []
    train_cost = []
    val_loss = []
    val_cost = []
    acc_train = []
    acc_val = []
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
            P_batch, fp_batch = ApplyNetwork(X_batch, trained_net)
            grads = BackwardPass(X_batch, Y_batch, P_batch, fp_batch, trained_net, lam)
            
            # Update weights
            for i in range(len(trained_net['W'])):
                trained_net['W'][i] -= eta * grads['W'][i]
                trained_net['b'][i] -= eta * grads['b'][i]
    
        P_train, _ = ApplyNetwork(X, trained_net)
        P_val, _ = ApplyNetwork(ValX, trained_net)
        
        tr_loss = ComputeLoss(P_train, y_labels)
        tr_cost = ComputeCost(P_train, y_labels, trained_net, lam)
        v_loss = ComputeLoss(P_val, val_y_labels)
        v_cost = ComputeCost(P_val, val_y_labels, trained_net, lam)
        
        tr_acc = float(ComputeAccuracy(P_train, y_labels).split()[0]) / 100
        v_acc = float(ComputeAccuracy(P_val, val_y_labels).split()[0]) / 100

        train_loss.append(tr_loss)
        train_cost.append(tr_cost)
        val_loss.append(v_loss)
        val_cost.append(v_cost)
        acc_train.append(tr_acc)
        acc_val.append(v_acc)
        update_steps.append(t) 
        
        #print(f"Epoch {epoch+1}: Cost={tr_cost:.4f}, Acc={v_acc:.4f}")
    
    return trained_net, train_loss, val_loss, train_cost, val_cost, acc_train, acc_val, update_steps

def CyclicLearningRate(eta_min, eta_max, t, n_s):
    # Implementation for cyclic learning rate
    l = t // (2 * n_s)

    if t <= (2*l +1) * n_s and t >= 2 * l * n_s:
        eta_t = eta_min + (t - 2 * l * n_s ) / n_s * (eta_max - eta_min)
    else:
        eta_t = eta_max - (t - (2 * l + 1) * n_s ) / n_s * (eta_max - eta_min)
    return eta_t

def VisualizeWeights(init_net):
    trained_net = copy.deepcopy(init_net)
    Ws = trained_net['W'].transpose().reshape((32, 32, 3, 10), order='F')
    W_im = np.transpose(Ws, (1, 0, 2, 3))
    fig=plt.figure()
    for i in range(10):
        w_im = W_im[:, :, :, i]
        w_im_norm = (w_im - np.min(w_im)) / (np.max(w_im) - np.min(w_im))
        plt.imsave(f"weight_visualization_class_{i}.png", w_im_norm)
        fig.add_subplot(1, 10, i+1)
    plt.imshow(w_im_norm)

def InitializeParameters(L, K, d, m, rng):
    net_params = {}
    net_params['W'] = [None] * L
    net_params['b'] = [None] * L

    for i in range(L):
        if i == 0:
            net_params['W'][i] = (1/np.sqrt(d))*rng.standard_normal(size=(m, d))
            net_params['b'][i] = np.zeros((m, 1))
        else:
            net_params['W'][i] = (1/np.sqrt(m))*rng.standard_normal(size=(K,m))
            net_params['b'][i] = np.zeros((K, 1))
    print(np.shape(net_params))
    return net_params

def gridSearchLambda(l_min, l_max, n_values=8):
    
    results = []

    for i in range(n_values):
        l = rng.uniform(l_min, l_max)
        lam = 10 ** l
        
        print(f"Testing value {i+1}/{n_values}: lambda = {lam:.4e}")

        trained_net, tr_c, val_c, tr_l, val_l = MiniBatchGD(
            TrainX_all, TrainY_all, Trainy_all, 
            ValX_5k, ValY_5k, Valy_5k, 
            GDperms, init_net, lam, rng
        )

        P_val, _ = ApplyNetwork(ValX_5k, trained_net)
        acc_str = ComputeAccuracy(P_val, Valy_5k)
        acc_val = float(acc_str.split()[0])

        result_entry = {
            "lambda": float(lam),
            "log_lambda": float(np.log10(lam)),
            "best_val_acc": acc_val,
            "final_val_loss": float(val_l[-1])
        }
        results.append(result_entry)

    results.sort(key=lambda x: x['best_val_acc'], reverse=True)

    with open('coarse_search_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nSearch complete. Results saved to 'coarse_search_results.json'")
    return results

X_all = []
Y_all = []
y_all = []

for i in range(1, 6):
    X, Y, y = LoadBatch(f'data_batch_{i}')
    X_all.append(X)
    Y_all.append(Y)
    y_all.append(y)

X_all = np.concatenate(X_all, axis=1)
Y_all = np.concatenate(Y_all, axis=1)   
y_all = np.concatenate(y_all, axis=0)

TrainX_all = X_all[:, :49000]
TrainY_all = Y_all[:, :49000]
Trainy_all = y_all[:49000]

TestX, TestY, Testy = LoadBatch('test_batch')

ValX_5k = X_all[:, 49000:]
ValY_5k = Y_all[:, 49000:]
Valy_5k = y_all[49000:]
        
"""
TrainX, TrainY, Trainy = LoadBatch('data_batch_1')
#print(TrainX.shape, TrainY.shape, Trainy.shape)
ValX, ValY, Valy = LoadBatch('data_batch_2')
TestX, TestY, Testy = LoadBatch('test_batch')
"""

mean_X, std_X = NormalizeData(TrainX_all)
TrainX = (TrainX_all - mean_X) / std_X
ValX = (ValX_5k - mean_X) / std_X
TestX = (TestX - mean_X) / std_X

K = TrainY_all.shape[0]
d = TrainX.shape[0]
rng = np.random.default_rng()
# get the BitGenerator used by default_rng
BitGen = type(rng.bit_generator)
# use the state from a fresh bit generator
seed = 42
rng.bit_generator.state = BitGen(seed).state


lam = 3.5532522663276575e-05
eta_min = 1e-5
eta_max = 1e-1
n_batch = 100
n = TrainX_all.shape[1]
n_s = 4 * np.floor(n / n_batch)
GDperms = {'n_batch': n_batch, 'n_epochs': 24, 'eta_min' : eta_min, 'eta_max': eta_max, 'n_s': n_s}
#X_small = TrainX[:,0:100]
#Y_small = TrainY[:, 0:100]
trained_params = InitializeParameters(2, 10, d, 50, rng)
init_net = copy.deepcopy(trained_params)

final_net, train_loss, val_loss, train_cost, val_cost, acc_train, acc_val,steps = MiniBatchGD(TrainX, TrainY_all, Trainy_all, ValX, ValY_5k, Valy_5k, GDperms, init_net, lam, rng)    

P_test, _ = ApplyNetwork(TestX, final_net)
test_acc = ComputeAccuracy(P_test, Testy)
print(f"--- FINAL TEST ACCURACY: {test_acc} ---")

#l_min = -6
#l_max =  -3
#randomSearchResults = gridSearchLambda(l_min, l_max, n_values=8)

"""
P_small_final, _ = ApplyNetwork(TrainX, trained_net)

loss_small = ComputeLoss(P_small_final, Trainy)
accuracy_small = ComputeAccuracy(P_small_final, Trainy)

#print("Loss: ", loss_small)
#print("Accuracy: ", accuracy_small)
"""
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.plot(steps, train_loss, 'g-', label='Training')
ax1.plot(steps, val_loss, 'r-', label='Validation')
ax1.set_ylim(0, max(max(train_loss), max(val_loss)) * 1.1)
ax1.set_title('Cost Function')
ax1.set_xlabel('Update step')
ax1.set_ylabel('Cost')
ax1.legend()

ax2.plot(steps, train_cost, 'g-', label='Training')
ax2.plot(steps, val_cost, 'r-', label='Validation')
ax2.set_ylim(0, max(max(train_cost), max(val_cost)) * 1.1)
ax2.set_title('Loss Function')
ax2.set_xlabel('Update step')
ax2.set_ylabel('Cost')
ax2.legend()

ax3.plot(steps,acc_train, 'g-', label='Training')
ax3.plot(steps, acc_val, 'r-', label='Validation')
ax3.set_ylim(0, max(max(acc_train), max(acc_val)) * 1.1)
ax3.set_title('Accuracy')
ax3.set_xlabel('Update step')
ax3.set_ylabel('Accuracy')
ax3.legend()

plt.tight_layout()
plt.show()

"""
d_small = 5
n_small = 3
m = 6
lam = 0

small_net = InitializeParameters(2, 10, d_small, m, rng)

X_small = TrainX[0:d_small, 0:n_small]
Y_small = TrainY[:, 0:n_small]
y_small = Trainy[0:n_small]

fp_data = ApplyNetwork(X_small, small_net)
P_small, fp_data = ApplyNetwork(X_small, small_net)

my_grads = BackwardPass(X_small, Y_small, P_small, fp_data, small_net, lam)
torch_grads = ComputeGradsWithTorch(X_small, y_small, small_net)

eps = 1e-10

diff_W1 = np.abs(my_grads['W'][0] - torch_grads['W'][0]) / max(eps, np.sum(np.sum(np.abs(my_grads['W'][0]) + np.abs(torch_grads['W'][0]), axis=0), axis=0))
diff_b1 = np.abs(my_grads['b'][0] - torch_grads['b'][0])/ max(eps, np.sum(np.abs(my_grads['b'][0]) + np.abs(torch_grads['b'][0])))

diff_W2 = np.abs(my_grads['W'][1] - torch_grads['W'][1]) / max(eps, np.sum(np.sum(np.abs(my_grads['W'][1]) + np.abs(torch_grads['W'][1]), axis=0), axis=0)) 
diff_b2 = np.abs(my_grads['b'][1] - torch_grads['b'][1])/ max(eps, np.sum(np.abs(my_grads['b'][1]) + np.abs(torch_grads['b'][1])))

print("Relative difference in W1: ", diff_W1)
print("Relative difference in b1: ", diff_b1)
print("Relative difference in W2: ", diff_W2)
print("Relative difference in b2: ", diff_b2)
"""




