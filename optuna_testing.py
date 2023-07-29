import optuna
from data_processing import create_dataloaders, load_data, process_data
from model import VAE
from loss_and_optimization import get_optimizer, get_scheduler, loss_function
from training_and_validation import train_model, validate_model
from visualization import visualize_and_save
import torch
import os

def objective(trial):
    # Downloading the dataset
    trainset, testset = load_data()

    # Preprocess Data 
    x_train, y_train, x_val, y_val = process_data(trainset, testset)

    # Suggested hyperparameters
    latent_dim = trial.suggest_int('latent_dim', 1, 10)
    batch_size = trial.suggest_int('batch_size', 16, 64, log=True)
    n_epochs = trial.suggest_int('n_epochs', 50, 80)

    # Fetch data loaders
    trainloader, testloader = create_dataloaders(x_train, y_train, x_val, y_val, batch_size=batch_size)

    # Instantiate the model
    model = VAE(latent_dim).to(device)

    # Define the optimizer and scheduler
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)

    # Get the checkpoint directory
    checkpoint_dir = './checkpts'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # If there's a checkpoint directory, try to resume.
    if checkpoint_dir is not None:
        checkpoint_path = os.path.join(checkpoint_dir, f'trial_{trial.number}.pt')

        # Try to load a saved checkpoint.
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            start_epoch = 0
    else:
        start_epoch = 0

     # Training loop
    for epoch in range(start_epoch, n_epochs):
        train_loss, err, kld = train_model(epoch, model, optimizer, trainloader, device)
        scheduler.step(train_loss)

        # If there's a checkpoint directory, save a checkpoint at the end of the epoch.
        if checkpoint_dir is not None:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)
    
    # Compute the validation loss
    test_loss = validate_model(model, testloader, device)

    # Visualize and save the input and output of the VAE from the validation set
    save_dir = '/pscratch/sd/g/gchen4/output_VAE_MNIST/figures'  # change this to your preferred directory
    visualize_and_save(model, testloader, device, save_dir)

    # The objective function needs to return the value it wants to minimize
    return test_loss

def main():
    study_name = 'VAE_Optuna_Testing'  # Define your study name
    
    # Fetch MySQL credentials from environment variables
    mysql_username = os.getenv('MYSQL_USERNAME')
    mysql_password = os.getenv('MYSQL_PASSWORD')
    mysql_host = os.getenv('MYSQL_HOST')
    mysql_db_name = os.getenv('MYSQL_DB_NAME')
    assert type(mysql_username) == str
    assert type(mysql_password) == str
    assert type(mysql_host) == str
    assert type(mysql_db_name) == str
    # Define your MySQL connection string using fetched credentials
    # storage_name = f'mysql://{mysql_username}:{mysql_password}@nerscdb04.nersc.gov/{mysql_db_name}'
    storage_name = 'sqlite:///example.db'
     # Try to load a saved study.
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
    except:
        # If the study does not exist, create a new one.
        study = optuna.create_study(study_name=study_name, storage=storage_name, direction='maximize')

    study.optimize(objective, n_trials=3) # you can adjust n_trials according to your requirement
    best_trial = study.best_trial

    print(f'Best trial: score {best_trial.value}, params {best_trial.params}')

if __name__ == "__main__":
    # Define the device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device is {device}')
    main()
