import torch
import torch.nn as nn
import numpy as np
import os
import time
from model.FESTGCN import FESTGCN
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from model.util import StandardScaler, metric, masked_mae, masked_mse
import yaml
from tqdm import tqdm

torch.set_num_threads(4)


def get_model(model_name, model_args):
    """Return the corresponding model instance based on the model name.
    
    Args:
        model_name: Name of the model.
        model_args: Dictionary of model parameters.
    
    Returns:
        model: Model instance.
    """
    if model_name.upper() == 'FESTGCN':
        return FESTGCN(
            adj=model_args['adj'],
            hidden_dim=model_args['hidden_dim'],
            scaler=model_args['scaler'],
            spectrum_similarity_matrix=model_args['spectrum_similarity_matrix'],
            time_delay_matrix=model_args['time_delay_matrix'],
            epsilon=model_args['epsilon'],  # New parameter
            sample_rate=model_args['sample_rate']  # New parameter
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

def load_data(data_path, ratio=1):
    """Load training, validation, and test data."""
    train_data = np.load(os.path.join(data_path, 'train_data.npz'))
    val_data = np.load(os.path.join(data_path, 'val_data.npz'))
    test_data = np.load(os.path.join(data_path, 'test_data.npz'))

    train_size = int(len(train_data['x']) * ratio)
    val_size = int(len(val_data['x']) * ratio)
    test_size = int(len(test_data['x']) * ratio)

    train_data = (train_data['x'][:train_size], train_data['y'][:train_size])
    val_data = (val_data['x'][:val_size], val_data['y'][:val_size])
    test_data = (test_data['x'][:test_size], test_data['y'][:test_size])

    return train_data, val_data, test_data

def create_data_loaders(train_data, val_data, test_data, batch_size):
    """Create DataLoader objects for training, validation, and test data."""
    train_dataset = TensorDataset(
        torch.FloatTensor(train_data[0]), 
        torch.FloatTensor(train_data[1])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_data[0]), 
        torch.FloatTensor(val_data[1])
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(test_data[0]), 
        torch.FloatTensor(test_data[1])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    # torch.autograd.set_detect_anomaly(True)
    
    model.train()
    total_loss = 0

    for iter, (x, y) in enumerate(tqdm(train_loader, desc="Training Progress")):

        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        
        output, average_time,grad_mask,baselines,log_pi,wait_penalty = model(x)
        label = y[:, 0, :]
        loss = masked_mae(output, label, 0)
        r = (2*(output.float().round().detach() == label.round()).float()-1)
        R = r.float() * grad_mask.float()
        b = grad_mask.float() * baselines.float()
        adjusted_reward = R - b.detach()
        loss_b = masked_mse(b, R, 0.0)
        loss_r = (-log_pi*adjusted_reward).sum(0).mean()
        lam = torch.tensor([0], dtype=torch.float, requires_grad=False).cuda(0)
        loss_all = 0.01 * loss_r + loss_b + loss + lam*wait_penalty
        
        loss_all.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device, mean, scale):
    """Validate the model."""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    total_average_time = 0
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validation Progress"):
            x, y = x.to(device), y.to(device)
            output, average_time,grad_mask,baselines,log_pi,wait_penalty = model(x)
            loss = masked_mae(output, y[:, 0, :], 0)
            total_loss += loss.item()

            predictions.append(output.cpu().numpy())
            targets.append(y[:, 0, :].cpu().numpy())

            total_average_time += average_time.item()
    
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Apply inverse transformation
    predictions = predictions * scale + mean
    targets = targets * scale + mean

    val_loss = total_loss / len(val_loader)
    predictions = torch.from_numpy(predictions)
    targets = torch.from_numpy(targets)
    mae, mape, rmse = metric(predictions,targets)
    apt = total_average_time/len(val_loader)*100

    return val_loss, mae, rmse, mape, apt

def test(model, test_loader, criterion, device, mean, scale):
    """Test the model with inverse transformation."""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    total_average_time = 0
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing Progress"):
            x, y = x.to(device), y.to(device)
            output, average_time,grad_mask,baselines,log_pi,wait_penalty = model(x)
            loss = masked_mae(output, y[:, 0, :], 0)
            total_loss += loss.item()
            
            predictions.append(output.cpu().numpy())
            targets.append(y[:, 0, :].cpu().numpy())

            total_average_time += average_time.item()
    
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    # Apply inverse transformation
    predictions = predictions * scale + mean
    targets = targets * scale + mean
    
    test_loss = total_loss / len(test_loader)
    predictions = torch.from_numpy(predictions)
    targets = torch.from_numpy(targets)
    mae, mape, rmse = metric(predictions,targets)
    apt = total_average_time/len(test_loader)*100
    
    return test_loss, mae, rmse, mape, apt

def main():
    with open('config.yaml') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    # Create save directory
    os.makedirs(config['data']['save_path'], exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load scaler parameters early
    scaler_data = np.load(os.path.join(config['data']['path'], 'scaler.npz'))
    mean, scale = scaler_data['mean'], scaler_data['std']
    
    # Load data
    train_data, val_data, test_data = load_data(config['data']['path'], config['data']['ratio'])
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data, config['model']['batch_size']
    )


    # adj_mx = pd.read_pickle(config['data']['distance_matrix'])[-1]  # Get adjacency matrix
    adj_mx = pd.read_pickle(config['data']['distance_matrix'])

    adj_spectrum = np.load(config['data']['spectrum_similarity_matrix'])

    time_delay_matrix = np.load(config['data']['time_delay_matrix'])

    # Create scaler
    scaler = StandardScaler(mean.astype(np.float32), scale.astype(np.float32))
    
    # Get model parameters
    num_nodes = train_data[0].shape[2]  # Number of nodes

    model_args = {
        'adj': adj_mx,  # Ensure it is float32 type
        'hidden_dim': config['model']['hidden_size'],
        'scaler': scaler,
        'spectrum_similarity_matrix': adj_spectrum,
        'time_delay_matrix': time_delay_matrix,
        'epsilon': config['model']['epsilon'],
        'sample_rate': config['model']['sample_rate']
    }
    
    # Initialize model and move to GPU
    model = get_model(config['model']['name'], model_args).to(device)
    
    # Ensure all model parameters are on GPU
    for param in model.parameters():
        param.data = param.data.to(device)
    
    # Loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['learning_rate'])
    
    # Training loop
    best_val_loss = float('inf')
    num_epochs = config['model']['num_epochs']
    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, mae, rmse, mape, apt = validate(model, val_loader, criterion, device, mean, scale)
        epoch_time = time.time() - start_time
        print(f'Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.2f}s)')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 
                      os.path.join(config['data']['save_path'], f"best_{config['model']['name'].lower()}.pth"))
    
    # Test best model
    model.load_state_dict(torch.load(os.path.join(config['data']['save_path'], f"best_{config['model']['name'].lower()}.pth")))
    test_loss, mae, rmse, mape, apt = test(model, test_loader, criterion, device, mean, scale)
    
    print('\nTest Results (After Inverse Transform):')
    print(f'Loss: {test_loss:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAPE: {mape:.2f}%')
    print(f'APT: {apt:.2f}%')

if __name__ == '__main__':
    main()