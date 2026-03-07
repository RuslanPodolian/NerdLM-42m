from nerdlm_app.model.transformer import DeepTransformer
from nerdlm_app.model.dataset import CustomDataset, DatasetPreparation
from nerdlm_app.model.training import TrainingEvaluating

import torch
from pathlib import Path
import glob
import os

from nerdlm_app.model.vocabulary import Vocabulary

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class NerdLM:
    def __init__(self, path=None, test_path=None, saved_model: bool = True, saved_model_name: str ='nerdlm_app/nerdlm.pt', training: bool = False, inference: bool = True):
        model_path = Path(saved_model_name)
        if not model_path.is_absolute():
            model_path = (PROJECT_ROOT / model_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.path = path
        self.model_path = model_path
        self.path_obj = None
        self.training = None


        if training:
            if self.path is not None:
                self.path_obj = Path(self.path)
                if not self.path_obj.is_absolute():
                    self.path_obj = (PROJECT_ROOT / self.path_obj).resolve()
                if self.path_obj.suffix.lower() != '.txt':
                    print("Sorry, only txt files are supported.")

                self.vocab_size = CustomDataset(str(self.path_obj)).get_word_map()

            self.training = TrainingEvaluating(str(self.path_obj), test_path)

            self.custom_dataset = CustomDataset(str(self.path_obj))
            self.word_map = self.custom_dataset.dataset_preparation.vocabulary.word_map
            print(f"Vocabulary size: {self.vocab_size}")

            self.model = DeepTransformer(
                d_model=512,
                d_ff=2048,
                num_heads=8,
                num_layers_gru=2,
                vocab_size=self.vocab_size
            )
            if saved_model and model_path.is_file():
                if model_path.stat().st_size == 0:
                    print(f"Saved model file is empty at '{model_path}'. Initializing a new model.")
                    self.model_path = model_path
                else:
                    try:
                        state_dict = torch.load(str(model_path), map_location=device)
                        saved_vocab = state_dict['fc.bias'].shape[0]
                        if saved_vocab != self.vocab_size:
                            self.vocab_size = saved_vocab
                            self.model = DeepTransformer(d_model=512, d_ff=2048, num_heads=8, num_layers_gru=2, vocab_size=saved_vocab)
                        self.model.load_state_dict(state_dict)
                        print(f"Successfully loaded saved model from '{model_path}'.")
                    except (EOFError, RuntimeError, ValueError) as exc:
                        print(f"Failed to load saved model at '{model_path}': {exc}. Initializing a new model.")
            elif saved_model:
                print(f"Saved model not found at '{model_path}'. Initializing a new model.")


        if inference:
            self.predictor = TrainingEvaluating(path=path, saved_model_name=model_path, training=False)
        self.dataset_preparation = DatasetPreparation()

    def train(self, epochs=10, save_model: bool = True):
        self.training.train(epochs=epochs)
        if save_model:
            self.training.save_model(self.model, self.model_path)

    def large_train(self, paths: list, epochs=10, save_model: bool = True):
        len_paths = len(paths)
        for i, path in enumerate(paths):
            self.path = path
            print(f"Dataset: {path}, count: [{i}/{len_paths}]")
            print("-"*15, f"Training on: {path}", "-"*15)
            self.training = TrainingEvaluating(path, test_path=None)
            self.training.train(epochs=epochs)
            # print(f"{path} training complete. Metrics: {self.training.evaluate()}")
            print("-"*50)

            if save_model:
                self.training.save_model(self.model, self.model_path)


    def generate_answer(self, question, previous_questions: list, convert_indices_to_words=True):
        if not question:
            print("Please enter a question.")

        tokens = self.predictor.predict(question, context=previous_questions, convert_to_text=convert_indices_to_words)

        if convert_indices_to_words:
            text = ' '.join(tokens)
        else:
            text = tokens

        print(f"Answer: {text}")

        return text

if __name__ == "__main__":
    bot = NerdLM('./nerdlm_app/model/datasets/english_extended_qa.txt', training=True, inference=False, saved_model_name='nerdlm_app/nerdlm.pt') # Paste any random dataset path
    data_dir = './nerdlm_app/model/datasets/'
    files = glob.glob(os.path.join(data_dir, '*.txt'))

    print(files)
    bot.large_train(files, epochs=30)
    # bot.train(epochs=100)