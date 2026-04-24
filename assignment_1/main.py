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
    """"
    s = np.matmul(W, X) + b
    P = np.exp(s) / np.sum(np.exp(s), axis=0, keepdims=True)
    """
    s = np.matmul(W, X) + b
    s = s - np.max(s, axis=0, keepdims=True)
    P = np.exp(s) / np.sum(np.exp(s), axis=0, keepdims=True)
    return P

def ComputeLoss(P, y):
    nn = P.shape[1]
    y_idx = y.astype(int)
    correct_class_probs = P[y_idx, np.arange(nn)]
    loss = np.mean(-np.log(correct_class_probs + 1e-10)) 
    return loss

def ComputeCost(P, y, network, lam):
    loss = ComputeLoss(P, y)
    W = network['W']
    cost = loss + lam * np.sum(W ** 2)
    return cost

def ComputeAccuracy(P, y):
    acc = 0
    predicted_classes = np.argmax(P, axis=0)
    real_classes = y
    acc = np.mean(predicted_classes == real_classes) * 100
    return str(acc) + " %"

def BackwardPass(X, Y, P, network, lam):
    W = network['W']
    b = network['b']
    n = X.shape[1]

    #Backward pass
    grad_W = (1/n) * (P - Y) @ X.T + 2 * lam * W 
    grad_b = (1/n) * np.sum(P - Y, axis=1, keepdims=True)
    grads = {'W': grad_W, 'b': grad_b}

    return grads

def MiniBatchGD(X, Y, y_labels, ValX, val_y_labels, GDperms, init_net, lam, rng):
    
    trained_net = copy.deepcopy(init_net)
    
    n = X.shape[1]
    n_batch = GDperms['n_batch']
    eta = GDperms['eta']
    n_epochs = GDperms['n_epochs']

    cost = ComputeCost(ApplyNetwork(X, trained_net), Y, trained_net, lam)
    #print("Initial cost: ", cost)
    train_cost = []
    val_cost = []
    train_loss = []
    val_loss = []

    for epoch in range(n_epochs):
        indices = rng.permutation(n)  
        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch
            inds = indices[j_start: j_end]
            X_batch = X[:, inds]
            Y_batch = Y[:, inds]
            P_batch = ApplyNetwork(X_batch, trained_net)
            grads = BackwardPass(X_batch, Y_batch, P_batch, trained_net, lam)
            
            trained_net['W'] = trained_net['W'] - eta * grads['W']
            trained_net['b'] = trained_net['b'] - eta * grads['b']
            
        P_train = ApplyNetwork(X, trained_net)
        P_val = ApplyNetwork(ValX, trained_net)
        
        train_cost.append(ComputeCost(P_train, y_labels, trained_net, lam))  
        val_cost.append(ComputeCost(P_val, val_y_labels, trained_net, lam))
        train_loss.append(ComputeLoss(P_train, y_labels))
        val_loss.append(ComputeLoss(P_val, val_y_labels))   
        
        #print(f"Epoch: {epoch} | Train Cost: {t_cost:.4f} | Val Cost: {v_cost:.4f}")
    return trained_net, train_cost, val_cost, train_loss, val_loss

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

def PlotResults(train_loss, val_loss, train_costs, val_costs, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(train_loss, 'g-', label='Training Loss')
    ax1.plot(val_loss, 'r-', label='Validation Loss')
    ax1.set_title(title + ' - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(train_costs, 'g-', label='Training Cost')
    ax2.plot(val_costs, 'r-', label='Validation Cost')
    ax2.set_title(title + ' - Cost')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Cost')
    ax2.legend()

    plt.tight_layout()
    plt.show()



TrainX, TrainY, Trainy = LoadBatch('data_batch_1')
#print(TrainX.shape, TrainY.shape, Trainy.shape)
ValX, ValY, Valy = LoadBatch('data_batch_2')

TestX, TestY, Testy = LoadBatch('test_batch')

mean_X, std_X = NormalizeData(TrainX)
TrainX = (TrainX - mean_X) / std_X
ValX = (ValX - mean_X) / std_X
TestX = (TestX - mean_X) / std_X

K = TrainY.shape[0]
d = TrainX.shape[0]
rng = np.random.default_rng()
# get the BitGenerator used by default_rng
BitGen = type(rng.bit_generator)
# use the state from a fresh bit generator
seed = 42
rng.bit_generator.state = BitGen(seed).state
init_net = {}
init_net['W'] = .01*rng.standard_normal(size = (K, d))
init_net['b'] = np.zeros((K, 1))

P = ApplyNetwork(TrainX[:, 0:100], init_net)
L = ComputeLoss(P, Trainy[0:100])
#print(L)
accuracy = ComputeAccuracy(P, Trainy[0:100])
#print("Train accuracy: ", accuracy)
#print(Trainy[0:100])
lam = 1.0

"""
my_grads = BackwardPass(TrainX[:, 0:100], TrainY[:,0:100], P, init_net, lam)
torch_grads = ComputeGradsWithTorch(TrainX[:, 0:100], Trainy[0:100], init_net)
eps = 1e-10

diff_W = np.abs(my_grads['W'] - torch_grads['W']) / max(eps, np.sum(np.sum(np.abs(my_grads['W']) + np.abs(torch_grads['W']), axis=0), axis=0))
diff_b = np.abs(my_grads['b'] - torch_grads['b'])/ max(eps, np.sum(np.abs(my_grads['b']) + np.abs(torch_grads['b'])))
#print("Relative difference in W: ", np.max(diff_W))
#print("Relative difference in b: ", np.max(diff_b))
"""

n_batch = 100
GDperms = {'n_batch': n_batch, 'eta': 0.001, 'n_epochs': 40}
trained_net, train_costs, val_costs, train_loss, val_loss = MiniBatchGD(TrainX, TrainY, Trainy, ValX, Valy, GDperms, init_net, lam, rng)

results = PlotResults(train_loss, val_loss, train_costs, val_costs, "($\eta=0.001, \lambda=1.0$)")
"""
plt.figure(figsize=(8, 6))
plt.plot(range(len(train_costs)), train_costs, 'g-', label='training loss')   # Green for training
plt.plot(range(len(val_costs)), val_costs, 'r-', label='validation loss') # Red for validation

plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right', frameon=True, shadow=True)
plt.grid(True, alpha=0.3)
plt.ylim(1.5, 2.1) 
plt.xlim(0, len(train_costs) - 1)
plt.tight_layout()

plt.show()
"""

P_test = ApplyNetwork(TestX, trained_net)
test_acc = ComputeAccuracy(P_test, Testy)
#print("Test accuracy: ", test_acc)

#learnt_weights = VisualizeWeights(trained_net)








