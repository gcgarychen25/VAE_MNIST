import optuna
from data_processing import create_dataloaders, load_data, process_data
from model import VAE
from loss_and_optimization import get_optimizer, get_scheduler, loss_function
from training_and_validation import train_model, validate_model
from visualization import visualize_and_save
import torch

def objective(trial):

    # Downloading the dataset
    trainset, testset = load_data()

    # Preprocess Data 
    x_train, y_train, x_val, y_val = process_data(trainset, testset)

    # Suggested hyperparameters
    latent_dim = trial.suggest_int('latent_dim', 1, 10)
    batch_size = trial.suggest_int('batch_size', 16, 64, log=True)
    n_epochs = trial.suggest_int('n_epochs', 5, 20)

    # Fetch data loaders
    trainloader, testloader = create_dataloaders(x_train, y_train, x_val, y_val, batch_size=batch_size)

    # Instantiate the model
    model = VAE(latent_dim).to(device)

    # Define the optimizer and scheduler
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)

    # Training loop
    for epoch in range(n_epochs):  
        train_loss, err, kld = train_model(epoch, model, optimizer, trainloader, device)
        scheduler.step(train_loss)
    
    # Compute the validation loss
    test_loss = validate_model(model, testloader, device)

    # Visualize and save the input and output of the VAE from the validation set
    save_dir = '/pscratch/sd/g/gchen4/output_VAE_MNIST/figures'  # change this to your preferred directory
    visualize_and_save(model, testloader, device, save_dir)

    # The objective function needs to return the value it wants to minimize
    return test_loss

def main():
    study = optuna.create_study(direction='maximize') # test loss is negative so we want to maximize it
    '''
    study = optuna.create_study(
        study_name='VAE-Distributed',
        storage='mysql://root@localhost/{MySQL name}',
        direction='maximize'
        )
    '''
    study.optimize(objective, n_trials=5) # you can adjust n_trials according to your requirement
    best_trial = study.best_trial

    print(f'Best trial: score {best_trial.value}, params {best_trial.params}')

if __name__ == "__main__":
    # Define the device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device is {device}')
    main()
