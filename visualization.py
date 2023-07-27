import matplotlib.pyplot as plt
import torch
import os

def visualize_and_save(model, testloader, device, save_dir):
    model.eval()
    data, _ = next(iter(testloader))
    data = data.to(device)
    with torch.no_grad():
        recon, _, _ = model(data)

    # detach data from GPU
    data = data.detach().cpu()
    recon = recon.detach().cpu()

    fig, ax = plt.subplots(2, 10, figsize=(20, 5))

    # plot original images
    for i in range(10):
        ax[0, i].imshow(data[i].reshape(14, 14), cmap='gray')
        ax[0, i].axis('off')

    # plot reconstructed images
    for i in range(10):
        ax[1, i].imshow(recon[i].reshape(14, 14), cmap='gray')
        ax[1, i].axis('off')

    # save figure to the specified directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, 'vae_visualization.png'))
