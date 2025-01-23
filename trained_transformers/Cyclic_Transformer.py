import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import random
import torch.onnx
from torch.onnx import export


# Semiautomaton definition
class CyclicSemiautomaton:
    def __init__(self, n):
        self.n = n
        self.states = list(range(n))  # Q = {0, 1, ..., n-1}
        self.sigma = list(range(n))  # S = {0, 1, ..., n-1}

    def transition(self, state, input_symbol):
        return (state + input_symbol) % self.n

    def compute_final_state(self, start_state, input_sequence):
        current_state = start_state
        for symbol in input_sequence:
            current_state = self.transition(current_state, symbol)
        return current_state


# Dataset generation function
def generate_dataset(num_samples=1000, n=5, max_seq_len=5):
    semiautomaton = CyclicSemiautomaton(n)
    data = []
    for i in range(num_samples):
        # Generate a random input sequence
        sequence_length = random.randint(1, max_seq_len)
        input_sequence = [random.choice(semiautomaton.sigma) for j in range(sequence_length)]
        # Compute the final state
        start_state = 0
        final_state = semiautomaton.compute_final_state(start_state, input_sequence)
        # Add to dataset
        data.append(("".join(map(str, input_sequence)), final_state))
    random.shuffle(data)
    df = pd.DataFrame(data, columns=["Input Sequence", "Final State"])
    return df


# Dataset class
class CyclicDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx, 0]
        tensor_input = torch.tensor([int(char) for char in sequence], dtype=torch.long)
        final_state = self.data.iloc[idx, 1]
        tensor_final = torch.tensor(final_state, dtype=torch.long)
        return tensor_input, tensor_final


# Function for padding sequences
def collate_fn(batch):
    inputs, labels = zip(*batch)
    max_len = max(len(x) for x in inputs)
    padded_inputs = torch.full((len(inputs), max_len), -1, dtype=torch.long)
    for i, seq in enumerate(inputs):
        padded_inputs[i, :len(seq)] = seq
    return padded_inputs, torch.tensor(labels)


# Transformer model
class CyclicTransformer(nn.Module):
    def __init__(self, num_tokens=2, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, max_seq_len=5,
                 num_states=5):
        super(CyclicTransformer, self).__init__()
        self.embedding = nn.Embedding(num_tokens, d_model, padding_idx=-1)
        self.pos_encoder = nn.Embedding(max_seq_len, d_model)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, num_states)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_key_padding_mask=None):
        padding_mask = src != -1
        src = src * padding_mask
        seq_len = src.size(1)
        pos = torch.arange(0, seq_len, device=src.device).unsqueeze(0)
        embeddings = self.embedding(src * padding_mask)
        pos_encoded = self.pos_encoder(pos)
        src_encoded = embeddings + pos_encoded * padding_mask.unsqueeze(-1)
        transformer_out = self.transformer_encoder(src_encoded.permute(1, 0, 2), src_key_padding_mask=src_key_padding_mask)
        out = self.fc_out(transformer_out[-1])
        return out


# Padding mask
def create_padding_mask(inputs, padding_value=-1):
    padding_mask = (inputs == padding_value)
    return padding_mask


# Training loop
def train_model(model, train_loader, test_loader, num_epochs=50, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            padding_mask = create_padding_mask(inputs)
            optimizer.zero_grad()
            outputs = model(inputs, src_key_padding_mask=padding_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
        evaluate_model(model, test_loader)


# Evaluation
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            padding_mask = create_padding_mask(inputs)
            outputs = model(inputs, src_key_padding_mask=padding_mask)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}%")


# Save dataset and transformer predictions
def save_datasets(dataframe, train_dataset, test_dataset, model, test_loader, file_prefix="semiautomaton"):
    # Save training data
    train_data = dataframe.iloc[train_dataset.indices]
    train_data.to_csv(f"{file_prefix}_train.csv", index=False)
    # Save test data
    test_data = dataframe.iloc[test_dataset.indices]
    test_data.to_csv(f"{file_prefix}_test.csv", index=False)
    # Predict on the test data
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            padding_mask = create_padding_mask(inputs)
            outputs = model(inputs, src_key_padding_mask=padding_mask)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
    test_data["Predicted Final State"] = predictions
    test_data.to_csv(f"{file_prefix}_test_predictions.csv", index=False)


# ONNX export function
def export_to_onnx(model, onnx_file_path="transformer_model.onnx", input_size=(1, 5), num_tokens=5, device='cpu'):
    dummy_input = torch.randint(0, num_tokens, input_size).to(device)
    # Padding mask for the dummy input
    padding_mask = create_padding_mask(dummy_input)
    model.eval()  # Set the model to evaluation mode
    torch.onnx.export(
        model,
        (dummy_input, padding_mask),  # Model inputs
        onnx_file_path,
        export_params=True,  # Trained parameter weights
        opset_version=14,
        do_constant_folding=True,
        input_names=["input", "padding_mask"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "sequence_length"},
            "padding_mask": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size"}})
    print(f"Model successfully exported to {onnx_file_path}")


if __name__ == "__main__":
    # Hyperparameters
    num_tokens = 5
    d_model = 64
    nhead = 4
    num_layers = 2
    dim_feedforward = 128
    max_seq_len = 5
    num_states = 5
    batch_size = 32
    num_epochs = 100  # 99.52% accuracy with 100 epochs for len = 20, mod 5, 150000 samples
    learning_rate = 1e-4
    # Generate dataset
    dataframe = generate_dataset(num_samples=1000, n=num_states)
    # Validate dataset
    # assert dataframe['Final State'].min() >= 0 and dataframe['Final State'].max() < num_states, "Final state labels are out of range!"
    # Train/test split
    dataset = CyclicDataset(dataframe)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize
    model = CyclicTransformer(num_tokens, d_model, nhead, num_layers, dim_feedforward, max_seq_len, num_states).to(
        device)
    # Train
    train_model(model, train_loader, test_loader, num_epochs, learning_rate)
    # Save dataset
    save_datasets(dataframe, train_dataset, test_dataset, model, test_loader, file_prefix="semiautomaton")
    # Export trained model
    export_to_onnx(
        model,
        onnx_file_path="cyclic_transformer.onnx",
        input_size=(1, max_seq_len),
        num_tokens=num_tokens,
        device=device)
