from data_processing import create_dataloaders, load_data, process_data
from model import VAE
from loss_and_optimization import get_optimizer, get_scheduler, loss_function
from training_and_validation import train_model, validate_model
from visualization import visualize_and_save
import torch

def main():
    # Define the device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device is {device}')

    # Downloading the dataset
    trainset, testset = load_data()

    # Preprocess Data 
    x_train, y_train, x_val, y_val = process_data(trainset, testset)
    
    # Fetch data loaders
    trainloader, testloader = create_dataloaders(x_train, y_train, x_val, y_val, batch_size=32)

    # Define latent dimension
    latent_dim = 2 # You can change this value

    # Instantiate the model
    model = VAE(latent_dim).to(device)

    # Define the optimizer and scheduler
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)

    # Training loop
    err_l, kld_l, n_wu, testl, update = [], [], [], [], []
    for epoch in range(2):  # change this to your desired number of epochs
        train_loss, err, kld = train_model(epoch, model, optimizer, trainloader, device)
        err_l.append(err)
        kld_l.append(kld)
        scheduler.step(train_loss)
        test_loss = validate_model(model, testloader, device)
        testl.append(test_loss)
    
    # Visualize and save the input and output of the VAE from the validation set
    save_dir = '/pscratch/sd/g/gchen4/output_VAE_MNIST/figures'  # change this to your preferred directory
    visualize_and_save(model, testloader, device, save_dir)

if __name__ == "__main__":
    main()
