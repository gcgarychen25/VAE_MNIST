import torch
from loss_and_optimization import loss_function

def train_model(epoch, model, optimizer, trainloader, device):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(trainloader):
        data = data.to(device)
        bsize = data.shape[0]
        recon_batch, mu, std = model(data)
        loss, err, kld = loss_function(recon_batch, data, mu, std)
        loss.backward()
        train_loss += err.item() + kld.item()
        optimizer.step()
        optimizer.zero_grad()

    average_train_loss = train_loss / len(trainloader.dataset)
    print('====> Epoch: {} Average train loss: {:.4f}'.format(epoch, average_train_loss))
    return average_train_loss, err.item()/bsize, kld.item()/bsize

def validate_model(model, testloader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in testloader:
            data = data.to(device)
            recon, mu, std = model(data)
            loss, err, kld = loss_function(recon, data, mu, std)
            test_loss += err + kld

    average_test_loss = test_loss / len(testloader.dataset)
    print('====> Test set loss: {:.4f}'.format(average_test_loss))
    return average_test_loss
