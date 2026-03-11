from nerdlm_app.model.transformer import DeepTransformer
from nerdlm_app.model.dataset import CustomDataset, DatasetPreparation

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

import logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

    def forward(self, prediction, target, mask=None):
        # Get original batch size and sequence length from prediction
        batch_size, seq_len, vocab_size = prediction.size()

        # Ensure target matches prediction sequence length
        if target.size(1) > seq_len:
            target = target[:, :seq_len]
        elif target.size(1) < seq_len:
            # Pad target if it's shorter (shouldn't normally happen)
            padding = torch.zeros(batch_size, seq_len - target.size(1), dtype=target.dtype, device=target.device)
            target = torch.cat([target, padding], dim=1)

        prediction = prediction.view(-1, vocab_size)
        target = target.contiguous().view(-1)

        if mask is None:
            # Mask out padding tokens in the target.
            mask = target.ne(0)
        mask = mask.float()
        mask = mask.reshape(-1) # do not use view -1, it returns an error

        labels = prediction.data.clone()
        labels.fill_(self.smooth / (self.size - 1))
        labels.scatter_(1, target.data.unsqueeze(1).long(), self.confidence)
        loss = self.criterion(prediction, labels)
        loss = (loss.sum(1) * mask).sum() / mask.sum()

        return loss

class TrainingEvaluating:
    def __init__(self, path=None, test_path=None, saved_model=True, saved_model_name='nerdlm_app/nerdlm.pt', training=True):
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.device = device

        self.dataset_preparation = DatasetPreparation()
        self.vocab_size = self.dataset_preparation.vocabulary.word_map.get('word_map_coef', len(self.dataset_preparation.vocabulary.word_map))

        if training:
            self.path = path

            self.path_obj = Path(self.path)
            if not self.path_obj.is_absolute():
                self.path_obj = (PROJECT_ROOT / self.path_obj).resolve()
            if self.path_obj.suffix.lower() != '.txt':
                print("Sorry, only txt files are supported.")

            self.custom_dataset = CustomDataset(str(self.path_obj))
            if test_path is None:
                self.test_dataset = self.custom_dataset
            else:
                self.test_dataset = CustomDataset(test_path)

            self.vocab_size = self.custom_dataset.get_word_map()
            self.custom_dataloader = DataLoader(self.custom_dataset, batch_size=32, shuffle=True)
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=32)

        self.model = DeepTransformer(
            d_model=512,
            d_ff=2048,
            num_heads=8,
            num_layers_gru=2,
            vocab_size=self.vocab_size
        )
        self.deep_transformer = self.model

        if saved_model_name:
            model_path = Path(saved_model_name)
            if not model_path.is_absolute():
                model_path = (Path(__file__).resolve().parents[2] / model_path).resolve()
            if model_path.is_file():
                if model_path.stat().st_size == 0:
                    logging.warning(f"Saved model file is empty at '{model_path}'. Using an untrained model.")
                else:
                    try:
                        state_dict = torch.load(str(model_path), map_location=device)
                        saved_vocab = state_dict['fc.bias'].shape[0]
                        if saved_vocab != self.vocab_size:
                            self.vocab_size = max(saved_vocab, self.vocab_size)
                            self.model = DeepTransformer(d_model=512, d_ff=2048, num_heads=8, num_layers_gru=2, vocab_size=self.vocab_size)
                            self.deep_transformer = self.model
                            # Copy saved weights into the (possibly larger) model
                            model_dict = self.deep_transformer.state_dict()
                            for key in state_dict:
                                if key in model_dict:
                                    if state_dict[key].shape == model_dict[key].shape:
                                        model_dict[key] = state_dict[key]
                                    else:
                                        # Partial copy for resized layers (embedding, fc)
                                        slices = tuple(slice(0, min(s, m)) for s, m in zip(state_dict[key].shape, model_dict[key].shape))
                                        model_dict[key][slices] = state_dict[key][slices]
                            self.deep_transformer.load_state_dict(model_dict)
                        else:
                            self.deep_transformer.load_state_dict(state_dict)
                    except (EOFError, RuntimeError, ValueError) as exc:
                        logging.warning(
                            f"Failed to load saved model at '{model_path}': {exc}. Using an untrained model."
                        )
            else:
                logging.warning(f"Saved model not found at '{model_path}'. Using an untrained model.")

        if training:
            self.adam_optimizer = optim.Adam(self.deep_transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
            self.transformer_optimizer = AdamWarmup(self.deep_transformer.d_model, warmup_steps=4000, optimizer=self.adam_optimizer)
            self.criterion = LossWithLS(self.custom_dataset.get_word_map(), smooth=0.1)
        else:
            self.deep_transformer.eval()

    def train(self, epochs):
        sum_loss = 0
        count = 0
        self.deep_transformer.train()

        print("Training started...")

        # print("X dataset tokens: ", self.custom_dataset.x)
        # print("Y dataset tokens: ", self.custom_dataset.y)
        # print(f"X size: {len(self.custom_dataset.x)}")
        # print(f"Y size: {len(self.custom_dataset.y)}")

        for epoch in range(epochs):
            for x, y in self.custom_dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                predictions, predictions_mask = self.deep_transformer(x)
                loss = self.criterion(predictions, y)
                self.adam_optimizer.zero_grad()
                loss.backward()
                self.transformer_optimizer.step()

                sum_loss += loss.item()
                count += 1

            print(f"Epoch: {epoch+1}/{epochs}; Current Loss: {loss.item()}; Total Loss: {sum_loss/count}")

        print("Training completed...")

    def save_model(self, model: nn.Module, path):
        torch.save(model.state_dict(), path)
        print("Model saved")

    def evaluate(self):
        self.deep_transformer.eval()
        sequences = []
        sum_loss = 0

        print("Evaluating started...")

        with torch.no_grad():
            for x, y in self.test_dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                predictions, predictions_mask = self.deep_transformer(x)
                loss = self.criterion(predictions, y)
                indices = torch.argmax(predictions, dim=-1)

                for index in indices:
                    sequences.append(self.custom_dataset.get_word_map()[index])
                sum_loss += loss.item()
                logging.info(f"Test Loss: {sum_loss/len(sequences)}")

        print("Evaluating completed...")

        return sequences

    def predict(self, text, context: list, convert_to_text=True):
        tokens = self.dataset_preparation.convert_line_to_tensor(text, expand_vocab=False)

        model_context = []
        for sentence in context:
            context = self.dataset_preparation.convert_line_to_tensor(sentence, expand_vocab=False)
            model_context.append(context)

        out = self.deep_transformer(tokens, model_context)
        indices = torch.argmax(out[0], dim=1)[0] # second one in out is list like [[True, False], [False, False]], and indices have extrac brackets

        if convert_to_text:
            output = []
            word_map = self.dataset_preparation.vocabulary.word_map
            for index in indices.data:
                skip_tokens = [word_map['<start>'], word_map['<pad>'], word_map['<unk>']]
                if 'ques' in word_map:
                    skip_tokens.append(word_map['ques'])
                if 'ans' in word_map:
                    skip_tokens.append(word_map['ans'])
                if index in skip_tokens:
                    continue

                elif index == word_map['<end>']:
                    break
                elif int(index) in self.dataset_preparation.vocabulary.idx_to_word:
                    output.append(self.dataset_preparation.vocabulary.idx_to_word[int(index)])
        else:
            output = indices

        return output

if __name__ == "__main__":
    training = TrainingEvaluating('datasets/english_extended_qa.txt', None)
    training.train(epochs=10)
