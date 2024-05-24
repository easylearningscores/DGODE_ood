import scipy.io
import torch
import torch.utils.data

def load_navier_stokes_data(path, sub=1, T_in=10, T_out=10, batch_size=20, reshape=None):
    ntrain = 1000
    neval = 100
    ntest = 100
    total = ntrain + neval + ntest
    f = scipy.io.loadmat(path)
    data = f['u'][..., 0:total]
    data = torch.tensor(data, dtype=torch.float32)

    # Training data
    train_a = data[:ntrain, ::sub, ::sub, :T_in]
    train_u = data[:ntrain, ::sub, ::sub, T_in:T_out+T_in] # [N, H,W,T]
    train_a = train_a.unsqueeze(-1).permute(0, 3, 1, 2, 4).permute(0, 1, 4, 2, 3)  # From [N, H, W, T] to [N, T, H, W, C]
    train_u = train_u.unsqueeze(-1).permute(0, 3, 1, 2, 4).permute(0, 1, 4, 2, 3)
    #print(train_a.shape, train_u.shape)
    # Evaluation data
    eval_a = data[ntrain:ntrain + neval, ::sub, ::sub, :T_in]
    eval_u = data[ntrain:ntrain + neval, ::sub, ::sub, T_in:T_out+T_in]
    eval_a = eval_a.unsqueeze(-1).permute(0, 3, 1, 2, 4).permute(0, 1, 4, 2, 3)  # From [N, H, W, T] to [N, T, H, W, C]
    eval_u = eval_u.unsqueeze(-1).permute(0, 3, 1, 2, 4).permute(0, 1, 4, 2, 3)
    # Testing data
    test_a = data[ntrain + neval:ntrain + neval + ntest, ::sub, ::sub, :T_in]
    test_u = data[ntrain + neval:ntrain + neval + ntest, ::sub, ::sub, T_in:T_out+T_in]
    test_a = test_a.unsqueeze(-1).permute(0, 3, 1, 2, 4).permute(0, 1, 4, 2, 3)  # From [N, H, W, T] to [N, T, H, W, C] to [N, T, C, H, W]
    test_u = test_u.unsqueeze(-1).permute(0, 3, 1, 2, 4).permute(0, 1, 4, 2, 3)

    if reshape:
        train_a = train_a.permute(reshape)
        train_u = train_u.permute(reshape)
        eval_a = eval_a.permute(reshape)
        eval_u = eval_u.permute(reshape)
        test_a = test_a.permute(reshape)
        test_u = test_u.permute(reshape)
        
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
    eval_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(eval_a, eval_u), batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

    return train_loader, eval_loader, test_loader