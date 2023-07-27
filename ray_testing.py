import ray
from ray import tune, air
from ray.air import session
from ray.tune.search.optuna import OptunaSearch
from data_processing import create_dataloaders, load_data, process_data
from model import VAE
from loss_and_optimization import get_optimizer, get_scheduler, loss_function
from training_and_validation import train_model, validate_model
from visualization import visualize_and_save
import torch

def objective(config):
    # Define the device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device is {device}')

    # Downloading the dataset
    trainset, testset = load_data()

    # Preprocess Data 
    x_train, y_train, x_val, y_val = process_data(trainset, testset)

    # Fetch data loaders
    batch_size = config["batch_size"]
    trainloader, testloader = create_dataloaders(x_train, y_train, x_val, y_val, batch_size=batch_size)

    # Instantiate the model
    latent_dim = config["latent_dim"]
    model = VAE(latent_dim).to(device)

    # Define the optimizer and scheduler
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)

    # Training loop
    n_epochs = config["n_epochs"]
    for epoch in range(n_epochs):  
        train_loss, err, kld = train_model(epoch, model, optimizer, trainloader, device)
        scheduler.step(train_loss)
    
    # Compute the validation loss
    test_loss = validate_model(model, testloader, device)
    session.report({"loss": test_loss})  

    # Visualize and save the input and output of the VAE from the validation set
    save_dir = '/pscratch/sd/g/gchen4/output_VAE_MNIST/figures'  # change this to your preferred directory
    visualize_and_save(model, testloader, device, save_dir)

    # The objective function needs to return the value it wants to minimize
    # tune.report(loss=test_loss)

def main():
    ray.init()

    '''# Define the hyperparameters search space
    search_space = {
        "latent_dim": tune.randint(1, 10),
        "batch_size": tune.randint(16, 64),
        "n_epochs": tune.randint(5, 50),
    }

    algo = OptunaSearch()

    tuner = tune.Tuner(
        objective,
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="max",
            search_alg=algo,
        ),
        run_config=air.RunConfig(
            num_samples=10),
        param_space=search_space,)
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)'''

    # Define the hyperparameters search space
    search_space = {
        "latent_dim": tune.randint(1, 10),
        "batch_size": tune.randint(16, 64),
        "n_epochs": tune.randint(5, 50),
    }

    algo = OptunaSearch(search_space)

    analysis = tune.run(
        objective,
        config=search_space,
        metric="loss",
        mode="max",
        search_alg=algo,
        num_samples=10,
    )
    
    print("Best config is:", analysis.get_best_config(metric="loss", mode="max"))


if __name__ == "__main__":
    main()
