import torch

def get_optimizer(model, lr=0.002):
    return torch.optim.Adam(model.parameters(), lr = lr)

def get_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, threshold=0.001, cooldown=0, min_lr=0.0001, verbose=True)

def loss_function(y, x, mu, std): 
    ERR = torch.nn.functional.binary_cross_entropy(y, x.view(-1, 196), reduction='sum')
    KLD = -0.5 * torch.sum(1 + torch.log(std**2) - mu**2 - std**2)
    return ERR + KLD, -ERR, -KLD
