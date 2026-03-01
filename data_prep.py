import torch
from tqdm import tqdm

def create_sequence(sequence_length, target_window, scaled_data):
    target_column_name = 'target'
    target_column_index = list(scaled_data.columns).index(target_column_name)
    x = []
    y = []
    for i in tqdm(range(len(scaled_data) - sequence_length - target_window + 1)):
        sequence = scaled_data.iloc[i:i+sequence_length, :].values
        target = scaled_data.iloc[i+sequence_length:i+sequence_length+target_window, target_column_index].values
        x.append(sequence)
        y.append(target)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)