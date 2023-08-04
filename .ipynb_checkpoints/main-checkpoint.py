from data_processing import create_dataloaders, load_data, process_data
from model import VAE
from loss_and_optimization import get_optimizer, get_scheduler, loss_function
from training_and_validation import train_model, validate_model
from visualization import visualize_and_save
import torch
import os
import argparse

def main(args):
    # Define the device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device is {device}')

    # Suggested hyperparameters
    latent_dim = args.latent_dim
    batch_size = args.batch_size
    n_epochs = args.epochs

    # Downloading the dataset
    trainset, testset = load_data()

    # Preprocess Data 
    x_train, y_train, x_val, y_val = process_data(trainset, testset)
    
    # Fetch data loaders
    trainloader, testloader = create_dataloaders(x_train, y_train, x_val, y_val, batch_size=batch_size)


    # Instantiate the model
    model = VAE(latent_dim).to(device)

    # Define the optimizer and scheduler
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)

    # Resume from a checkpoint if it exists
    start_epoch = 0
    checkpt_name = f'{args.trial_id}.pt'
    trial_dir = os.path.join(args.working_dir, 'trials', args.trial_id)
    if not os.path.exists(trial_dir):
        os.makedirs(trial_dir)
    checkpoint_path = os.path.join(trial_dir, checkpt_name)
    if os.path.exists(checkpoint_path):
        print('loading checkpoints')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # Training loop
    err_l, kld_l, n_wu, testl, update = [], [], [], [], []
    for epoch in range(start_epoch, n_epochs):  # change this to your desired number of epochs
        train_loss, err, kld = train_model(epoch, model, optimizer, trainloader, device)
        err_l.append(err)
        kld_l.append(kld)
        scheduler.step(train_loss)
        test_loss = validate_model(model, testloader, device)
        testl.append(test_loss)

        # Save model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, checkpoint_path)
    
    # Visualize and save the input and output of the VAE from the validation set
    figure_name = f'figures_{args.trial_id}.png'
    figure_dir = os.path.join(trial_dir, 'figures', figure_name)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    visualize_and_save(model, testloader, device, figure_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('encoder decoder examiner')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size per GPU')
    parser.add_argument('--epochs', type=int, default=100,
                        help='num of training epochs')
    parser.add_argument('--latent_dim', type=int, default=3,
                        help='latent space dimension')
    parser.add_argument('--trial_id', type=str, default=3,
                        help='trial_parameters')
    parser.add_argument('--working_dir', type=str, default=3,
                        help='working_dir')
    args = parser.parse_args()
    main(args)
