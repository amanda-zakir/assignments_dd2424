import torch
import numpy as np

def ComputeGradsWithTorch(X, y, MX, net_params, lam):
    
    MX_torch = torch.from_numpy(MX)
    n = X.shape[3]
    #print("MX_torch shape:", MX_torch.shape)
    #print("Torch patch:", MX_torch[0, :, 0][:10])

    Xt_numpy = X
    Xt = torch.from_numpy(Xt_numpy)
    #print("Xt shape:", Xt.shape)

    f = net_params['Fs'].shape[0]
    print(net_params['Fs'].shape)
    n_rows = Xt.shape[0]//f
    n_cols = Xt.shape[1]//f
    n_p = n_rows * n_cols
    nf = net_params['Fs'].shape[3]
    

    L = len(net_params['W'])

    # will be computing the gradient w.r.t. these parameters    
    W = [None] * L
    b = [None] * L   
    #print("W1 shape:", net_params['W'][0].shape)
    Fs_np = net_params['Fs']
    """
    
    Fs_flat = np.zeros((f*f*3, nf))

    for i in range(nf):
        Fs_flat[:, i] = Fs_np[:,:,:,i].reshape(-1, order='C')
        #Fs_flat[:, i] = Fs_np[:, :, :, i].transpose(0, 2, 1).reshape(-1)
    

    Fs = torch.from_numpy(Fs_flat).requires_grad_(True)
    """
    Fs_flat = Fs_np.reshape(f*f*3, nf, order='C')
    Fs = torch.from_numpy(Fs_flat).requires_grad_(True)
    #print("Fs shape:", Fs.shape)
    #print("Torch Fs shape:", net_params['Fs'].shape)
    #print("Torch Fs shape:", net_params['Fs'][0,0,0,:])

    for i in range(L):
        W[i] = torch.tensor(net_params['W'][i], requires_grad=True)
        b[i] = torch.tensor(net_params['b'][i], requires_grad=True)

    ## give informative names to these torch classes        
    apply_relu = torch.nn.ReLU()
    apply_softmax = torch.nn.Softmax(dim=0)

    #### BEGIN your code ###########################
    
    # Apply the scoring function corresponding to equations (1-3) in assignment description 
    # If X is d x n then the final scores torch array should have size 10 x n 
    conv_outputs = []

    for i in range(n):
       conv_outputs.append(MX_torch[:, :, i] @ Fs)
       #conv_outputs.append((Fs.T @ MX_torch[:, :, i].T).T)

    conv_outputs = torch.stack(conv_outputs, dim=2)
    #print("torch conv:", conv_outputs.shape)
    
    #h_conv = conv_outputs.reshape(n_p * nf, n)
    h_conv = conv_outputs.reshape(n_p * nf, n)
    h_conv = apply_relu(h_conv)
    #print("h_conv:", h_conv.shape)
    #print("W1 shape:", W[0].shape)
    
    s_1 = W[0] @ h_conv + b[0]
    H = apply_relu(s_1)
    scores = W[1] @ H + b[1]

    #### END of your code ###########################            

    # apply SoftMax to each column of scores     
    P = apply_softmax(scores)
    print("Torch P:", P[:5,0])

    
    # compute the loss
    y_torch = torch.from_numpy(y).long() 
    if y_torch.dim() > 1:
        y_torch = torch.argmax(y_torch, dim=0)
    loss = torch.mean(-torch.log(P[y_torch, torch.arange(n)]))
    #print("Torch n:", n)
    #print("Loss torch:", loss)

    reg = 0
    for i in range(L):
        reg += torch.sum(W[i] **2)
    reg += torch.sum(Fs ** 2)
    cost = loss + lam * reg

    # 4. Compute cross-entropy loss
    
    # compute the backward pass relative to the loss and the named parameters 
    loss.backward()

    # extract the computed gradients and make them numpy arrays 
    grads = {}
    grads['W'] = [None] * L
    grads['b'] = [None] * L
    grads['Fs'] = Fs.grad.numpy().reshape((4,4,3,2), order='C')
    for i in range(L):
        grads['W'][i] = W[i].grad.numpy()
        grads['b'][i] = b[i].grad.numpy()
    return grads
