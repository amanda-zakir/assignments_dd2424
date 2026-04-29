import torch
import numpy as np

def ComputeGradsWithTorch(X, y, MX, Fs, network_params, conv_flat):
    
    MX_torch = torch.from_numpy(MX).float()
    Xt = torch.from_numpy(X)
    n_p = MX.shape[0]
    nf = Fs.shape[3]
    n = X.shape[0]

    Fs_np = Fs.transpose(2, 0, 1, 3).reshape((-1, nf), order='C')
    Fs_flat = torch.tensor(Fs_np, dtype=torch.float, requires_grad=True)
    
    conv_flat = torch.tensor(conv_flat.reshape((-1, n), order='C'), 
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
    """
    h_conv = torch.zeros((n_p, nf, n), dtype=torch.float)
    
    for i in range(n):
       h_conv[:, :, i] = torch.matmul(MX_torch[:, :, i], Fs_flat)
    """
    conv_output = []
    for i in range(n):
        conv_output.append(torch.matmul(MX_torch[:, :, i], Fs_flat))
    
    h_conv = torch.stack(conv_output, dim=2)
    h_conv_relu = apply_relu(h_conv)
    h_permuted = h_conv_relu.permute(1, 0, 2) 
    h_flat = h_permuted.reshape(nf * n_p, n)
    H = torch.matmul(W[0], h_flat) + b[0]
    H = apply_relu(H)
    print("H sum:", H.sum().item())
    scores = torch.matmul(W[1], H) + b[1]
    print("Scores sum:", scores.sum().item())

    #### END of your code ###########################            

    # apply SoftMax to each column of scores     
    P = apply_softmax(scores)

    
    # compute the loss
    y_torch = torch.from_numpy(y).long() 
    print("Torch labels:     ", y_torch[:5].cpu().numpy())
    
    # 2. Get the actual number of images in this batch
    batch_size = P.shape[1] 
    
    # 3. Use torch.arange instead of np.arange to keep everything in Torch
    # This picks the correct class probability for every image in the batch
    correct_class_probs = P[y_torch, torch.arange(batch_size)]
    
    # 4. Compute cross-entropy loss
    loss = torch.mean(-torch.log(correct_class_probs))
    print("Loss torch:", loss.item())
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
