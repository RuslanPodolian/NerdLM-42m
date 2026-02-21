from model.structure.transformer import DeepTransformer
from model.preprocess.dataset_preparation import CustomDataset, DatasetPreparation

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

import logging

class AdamWarmup:
    def __init__(self, model_size, warmup_steps, optimizer):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.current_step = 0
        self.lr = 0

    def get_lr(self):
        return self.model_size ** (-0.5) * min(self.current_step ** (-0.5), self.current_step * self.warmup_steps ** (-1.5))

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.lr = lr

        self.optimizer.step()

class LossWithLS(nn.Module):
    def __init__(self, size, smooth):
        super(LossWithLS, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        self.smooth = smooth
        self.confidence = 1 - smooth
        self.size = size

    def forward(self, prediction, target, mask):
        prediction = prediction.view(-1, prediction.size(-1))
        target = target.view(-1)
        mask = mask.float()
        mask = mask.view(-1)
        labels = prediction.data.clone()
        labels.fill_(self.smooth / (self.size - 1))
        labels.scatter(1, target.data.unsqueeze(1), self.confidence)
        loss = self.criterion(prediction, labels)
        loss = (loss.sum(1) * mask).sum() / mask.sum()

        return loss

class TrainingEvaluating:
    def __init__(self, path, test_path, device='cpu'):
        if torch.cuda.is_available():
            device = 'cuda'

        self.custom_dataset = CustomDataset(path)
        if test_path is None:
            self.test_dataset = self.custom_dataset
        else:
            self.test_dataset = CustomDataset(test_path)
        self.custom_dataloader = DataLoader(self.custom_dataset, batch_size=32, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=32)

        self.deep_transformer = DeepTransformer(d_model=512, num_heads=8, num_layers_gru=2, vocab_size=self.custom_dataset.get_word_map(), d_ff=2048, dropout_rate=0.1).to(device)
        self.adam_optimizer = optim.Adam(self.deep_transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        self.transformer_optimizer = AdamWarmup(self.deep_transformer.d_model, warmup_steps=4000, optimizer=self.adam_optimizer)
        self.criterion = LossWithLS(self.custom_dataset.get_word_map(), smooth=0.1)

    def train(self, epochs):
        sum_loss = 0
        count = 0
        self.deep_transformer.train()

        print("X dataset tokens: ", self.custom_dataset.x)
        print("Y dataset tokens: ", self.custom_dataset.y)
        print(f"X size: {len(self.custom_dataset.x)}")
        print(f"Y size: {len(self.custom_dataset.y)}")

        for epoch in range(epochs):
            for x, y in self.custom_dataloader:
                x = x.to(self.deep_transformer.device)
                y = y.to(self.deep_transformer.device)

                predictions, predictions_mask = self.deep_transformer(x)
                loss = self.criterion(predictions, y, predictions_mask)
                self.transformer_optimizer.zero_grad()
                loss.backward()
                self.transformer_optimizer.step()

                sum_loss += loss.item()
                count += 1

                logging.info(f"Epoch: {epoch}/{epochs}; Current Loss: {loss.item()}; Total Loss: {sum_loss/count}")

    def save_model(self, path):
        torch.save(self.deep_transformer.state_dict(), path)

    def evaluate(self):
        self.deep_transformer.eval()
        sequences = []
        sum_loss = 0

        with torch.no_grad():
            for x, y in self.test_dataloader:
                x = x.to(self.deep_transformer.device)
                y = y.to(self.deep_transformer.device)

                predictions, _ = self.deep_transformer(x)
                loss = self.criterion(predictions, y)
                indices = torch.argmax(predictions, dim=-1)

                for index in indices:
                    sequences.append(self.custom_dataset.get_word_map()[index])
                sum_loss += loss.item()
                logging.info(f"Test Loss: {sum_loss/len(sequences)}")
        return sequences

class Predictor:
    def __init__(self, saved_model_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.deep_transformer = DeepTransformer(d_model=512, num_heads=8, num_layers_gru=2, vocab_size=10000).to(device)
        if saved_model_path:
            model_path = Path(saved_model_path)
            if not model_path.is_absolute():
                model_path = (Path(__file__).resolve().parents[2] / model_path).resolve()
            if model_path.is_file():
                if model_path.stat().st_size == 0:
                    logging.warning(f"Saved model file is empty at '{model_path}'. Using an untrained model.")
                else:
                    try:
                        self.deep_transformer.load_state_dict(torch.load(str(model_path), map_location=device))
                    except (EOFError, RuntimeError, ValueError) as exc:
                        logging.warning(
                            f"Failed to load saved model at '{model_path}': {exc}. Using an untrained model."
                        )
            else:
                logging.warning(f"Saved model not found at '{model_path}'. Using an untrained model.")
        self.dataset_preparation = DatasetPreparation()
        self.deep_transformer.eval()

    def predict(self, text):
        sequences = []
        tokens = self.dataset_preparation.convert_line_to_tensor(text)
        out = self.deep_transformer(tokens)
        indices = torch.argmax(out[0], dim=-1)[0] # second one in out is list like [[True, False], [False, False]], and indices have extrac brackets

        for index in indices:
            sequences.append(self.dataset_preparation.vocabulary.idx_to_word[int(index)])

        return sequences


if __name__ == "__main__":
    training = TrainingEvaluating('../../model/datasets/english_qa/extended_qa_dataset.txt', None)
    training.train(epochs=10)
