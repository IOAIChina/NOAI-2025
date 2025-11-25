import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

device = 'cuda'

with open("./train.json", "r") as f:
    data = list(json.load(f).items())

# Character vocabulary
chars = sorted(list(set("".join([word for word, _ in data]))))
char2idx = {char: idx + 1 for idx, char in enumerate(chars)}  # 0 is reserved for padding
idx2char = {idx: char for char, idx in char2idx.items()}
vocab_size = len(chars)


# Define Dataset
class CompoundDataset(Dataset):
    def __init__(self, data, char2idx):
        self.data = data
        self.char2idx = char2idx

    def __len__(self):
        return len(self.data)

    def encode(self, word, labels):
        return (
            torch.tensor([self.char2idx[char] for char in word], dtype=torch.long),
            torch.tensor(labels, dtype=torch.float),
        )

    def __getitem__(self, idx):
        word, labels = self.data[idx]
        return self.encode(word, labels)


# Collate function to handle batching
def collate_fn(batch):
    inputs, targets = zip(*batch)
    lengths = [len(seq) for seq in inputs]
    max_len = max(lengths)

    padded_inputs = torch.zeros(len(inputs), max_len, dtype=torch.long)
    padded_targets = torch.zeros(len(targets), max_len, dtype=torch.float)

    for i, (seq, tgt) in enumerate(zip(inputs, targets)):
        padded_inputs[i, : len(seq)] = seq
        padded_targets[i, : len(tgt)] = tgt

    return padded_inputs, padded_targets, lengths


# Define the BiLSTM Model
class BiLSTMSegmentation(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, num_layers=2):
        super(BiLSTMSegmentation, self).__init__()
        self.vocab_size = vocab_size
        self.lstm = nn.LSTM(
            vocab_size + 1,  # input_size is vocab_size + 1 (for one-hot encoding)
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)  # BiLSTM outputs are concatenated
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch_size, seq_length)
        # Convert input to one-hot encoding
        x = nn.functional.one_hot(x, num_classes=self.vocab_size + 1).float()  # (batch_size, seq_length, vocab_size + 1)
        
        # Pass through BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_length, hidden_dim * 2)
        
        # Apply a fully connected layer to get binary classification
        logits = self.fc(lstm_out)  # (batch_size, seq_length, 1)
        logits = logits.squeeze(-1)  # (batch_size, seq_length)
        return self.sigmoid(logits)  # (batch_size, seq_length)


def train():
    # Initialize Dataset and DataLoader
    dataset = CompoundDataset(data, char2idx)
    
    batch_size = 128
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, prefetch_factor=2
    )
    
    # Initialize Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMSegmentation(vocab_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    num_epochs = 32
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets, lengths in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Mask padding positions
            mask = torch.arange(inputs.shape[1])[None, :] < torch.tensor(lengths)[:, None]
            mask = mask.to(device)
            outputs = outputs[mask]
            targets = targets[mask]
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
    return model


def predict_and_save(model, input_file, output_file, char2idx, device="cpu"):
    """
    Reads a JSON file, predicts segmentation for each word, and saves the results to a new JSON file.

    :param model: Trained model
    :param input_file: Path to input JSON file
    :param output_file: Path to output JSON file
    :param char2idx: Character to index mapping
    :param device: Device to run the model on
    """
    # Load the input data
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Initialize predictions dictionary
    predictions = {}

    # Set model to evaluation mode
    model.eval()

    # Predict for each word
    with torch.no_grad():
        for word, _ in data.items():
            # Convert word to indices
            indices = [char2idx.get(char, 0) for char in word]
            input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
            # Get model outputs
            outputs = model(input_tensor)[0].cpu().numpy()
            
            # Convert outputs to binary labels
            boundaries = (outputs > 0.6).astype(int)
            predictions[word] = boundaries.tolist()

    # Save predictions to output file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)

    logging.info(f"Predictions saved to {output_file}")


# Example Usage
if __name__ == "__main__":
    # Train the model
    model = train()
    
    # Predict and save results
    input_file = "./val.json"
    output_file = "./submissionval.json"
    predict_and_save(model, input_file, output_file, char2idx, device)
     # Predict and save results
    input_file = "./test.json"
    output_file = "./submissiontest.json"
    predict_and_save(model, input_file, output_file, char2idx, device)

