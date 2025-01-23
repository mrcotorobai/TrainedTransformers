import random
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence


# Transition rules
def parity_automaton(sequence):
    state = 'even'  # Starting state
    for char in sequence:
        if state == 'even':
            state = 'even' if char == '0' else 'odd'
        elif state == 'odd':
            state = 'odd' if char == '0' else 'even'
    return state


# Dataset generation
def generate_dataset(num_samples=30000, max_length=15):
    data = []
    unique_sequences = set()
    while len(data) < num_samples:
        length = random.randint(1, max_length)
        sequence = ''.join(random.choice(['0', '1']) for i in range(length))
        if sequence not in unique_sequences:
            unique_sequences.add(sequence)
            final_state = parity_automaton(sequence)
            data.append((sequence, final_state))
    df = pd.DataFrame(data, columns=['Input Sequence', 'Final State'])
    return df


class ParityDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.statemap = {'odd': 1, 'even': 0}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sequence = self.data.iloc[item, 0]
        tensor_input = torch.tensor([int(char) for char in sequence], dtype=torch.long)
        final_state = self.statemap[self.data.iloc[item, 1]]
        tensor_final = torch.tensor(final_state, dtype=torch.long)
        return tensor_input, tensor_final


# Padding
def collate_fn(batch):
    inputs, labels = zip(*batch)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=-1)
    labels = torch.stack(labels)
    return padded_inputs, labels


# Transformer model
class ParityTransformer(nn.Module):
    def __init__(self, num_tokens=2, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, max_seq_len=20):
        super(ParityTransformer, self).__init__()

        self.embedding = nn.Embedding(num_tokens, d_model)
        self.pos_encoder = nn.Embedding(max_seq_len, d_model)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, 2)  # Output layer (for 2 classes: even and odd)

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
        transformer_out = self.transformer_encoder(src_encoded.permute(1, 0, 2),
                                                   src_key_padding_mask=src_key_padding_mask)
        out = self.fc_out(transformer_out[-1])
        return out


# Adjust the optimizer and learning rate
def train_model(model, train_loader, test_loader, num_epochs=50, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

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
            # print("Predicted: ", predicted.cpu().numpy())
            # print("Actual:    ", labels.cpu().numpy())

    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}%")


data = generate_dataset(num_samples=30000)
parity_data = ParityDataset(data)
train_size = int(0.8 * len(parity_data))
test_size = len(parity_data) - train_size
train_dataset, test_dataset = random_split(parity_data, [train_size, test_size])
# train_dataset.to_csv("train_data.csv", index=False)
# test_dataset.to_csv("test_data.csv", index=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ParityTransformer().to(device)
train_model(model, train_loader, test_loader, num_epochs=200, lr=1e-4)



