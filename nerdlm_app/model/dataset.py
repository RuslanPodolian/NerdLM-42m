import torch
from pathlib import Path
from torch.utils.data import Dataset
from nerdlm_app.model.vocabulary import Vocabulary


class DatasetPreparation:
    def __init__(self):
        self.vocabulary = Vocabulary()

    def convert_line_to_tensor(self, line, target: bool = True):
        tokens = line.split()

        self.vocabulary.add_word_per_line(tokens)

        padding = lambda base_sequence: [self.vocabulary.word_map['<pad>'] for _ in range(self.vocabulary.max_length-len(base_sequence))]

        if target:
            base_sequence = [self.vocabulary.word_map['<start>']] + [self.vocabulary.word_map.get(token, self.vocabulary.word_map['<unk>']) for token in tokens] + [self.vocabulary.word_map['<end>']]
            padded_sequence = padding(base_sequence)
            out_sequence = base_sequence + padded_sequence
        else:
            base_sequence = [self.vocabulary.word_map.get(token, self.vocabulary.word_map['<unk>']) for token in tokens]
            padded_sequence = padding(base_sequence)
            out_sequence = base_sequence + padded_sequence

        return torch.LongTensor(out_sequence)

    def remove_padding(self, tokens):
        prepared_tokens = []

        for token in tokens:
            if token == self.vocabulary.word_map['<pad>']:
                tokens.remove(token)
                prepared_tokens.append(token)
            else:
                prepared_tokens.append(token)

        return prepared_tokens

    def load_dataset(self, path):
        path = self._validate_path(path)
        lines = self.vocabulary.load_lines_of_text_file(path)
        self.vocabulary.add_word_per_line(lines)

        x = []
        y = []

        for line in lines:
            if 'ans:' in line:
                y.append(self.convert_line_to_tensor(line, target=True))
            elif 'ques:' in line:
                x.append(self.convert_line_to_tensor(line, target=False))

        return x, y

    @staticmethod
    def _validate_path(path):
        if path is None:
            raise ValueError("Dataset path is None. Provide a valid text file path.")
        path_obj = Path(path)
        if not path_obj.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path_obj}")
        return str(path_obj)

    def get_word_map(self):
        return len(self.vocabulary.word_map)


class CustomDataset(Dataset):
    def __init__(self, path):
        self.path = DatasetPreparation._validate_path(path)
        self.dataset_preparation = DatasetPreparation()
        self.x, self.y = self.dataset_preparation.load_dataset(self.path)

        print(f"Dataset loaded; vocabulary size: {self.dataset_preparation.get_word_map()}; x length: {len(self.x)}; y length: {len(self.y)}")

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def get_word_map(self):
        return self.dataset_preparation.get_word_map()

    def __len__(self):
        return len(self.x)
