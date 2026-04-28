import torch
import numpy as np

def ComputeGradsWithTorch(X, y, MX, Fs, network_params):
    
    MX_torch = torch.from_numpy(MX).float()
    Xt = torch.from_numpy(X)
    n_p = MX.shape[0]
    nf = Fs.shape[3]
    n = X.shape[3]

    Fs_flat = torch.tensor(Fs.reshape((-1, nf), order='C'), 
                           dtype=torch.float, requires_grad=True)

    L = len(network_params['W'])

    # will be computing the gradient w.r.t. these parameters    
    W = [None] * L
    b = [None] * L    
    for i in range(len(network_params['W'])):
        W = [torch.tensor(w, requires_grad=True).float() for w in network_params['W']]
        b = [torch.tensor(bi, requires_grad=True).float() for bi in network_params['b']]       

    ## give informative names to these torch classes        
    apply_relu = torch.nn.ReLU()
    apply_softmax = torch.nn.Softmax(dim=0)

    #### BEGIN your code ###########################
    
    # Apply the scoring function corresponding to equations (1-3) in assignment description 
    # If X is d x n then the final scores torch array should have size 10 x n 
    h_conv = torch.zeros((n_p, nf, n), dtype=torch.float)
    
    for i in range(n):
       h_conv[:, :, i] = torch.matmul(MX_torch[:, :, i], Fs_flat)
    
    h_conv_relu = apply_relu(h_conv)

    h_flat = h_conv_relu.reshape(n_p * nf, n)
    H = torch.matmul(W[0], h_flat) + b[0]
    apply_relu = torch.nn.ReLU()
    H = apply_relu(H)
    scores = torch.matmul(W[1], H) + b[1]

    #### END of your code ###########################            

    # apply SoftMax to each column of scores     
    P = apply_softmax(scores)
    
    # compute the loss
    print("h_flat shape:", h_flat.shape)
    y_torch = torch.from_numpy(y).long() 
    
    # 2. Get the actual number of images in this batch
    batch_size = P.shape[1] 
    
    # 3. Use torch.arange instead of np.arange to keep everything in Torch
    # This picks the correct class probability for every image in the batch
    correct_class_probs = P[y_torch, torch.arange(batch_size)]
    
    # 4. Compute cross-entropy loss
    loss = torch.mean(-torch.log(correct_class_probs))
    
    # compute the backward pass relative to the loss and the named parameters 
    loss.backward()

    # extract the computed gradients and make them numpy arrays 
    grads = {}
    grads['W'] = [None] * L
    grads['b'] = [None] * L
    grads['Fs_flat'] = [None]
    for i in range(L):
        grads['W'][i] = W[i].grad.detach().cpu().numpy()
        grads['b'][i] = b[i].grad.detach().cpu().numpy()
    grads['Fs_flat'] = Fs_flat.grad.detach().cpu().numpy()
    return grads
