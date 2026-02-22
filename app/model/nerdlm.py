from app.model.transformer import DeepTransformer
from app.model.dataset import CustomDataset
from app.model.training import TrainingEvaluating, Predictor
import torch
from pathlib import Path
import glob
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class NerdLM:
    def __init__(self, path=None, test_path=None, saved_model: bool = True, saved_model_name: str ='model/saved_models/nerdlm.pt', training: bool = False, inference: bool = True, device='cpu'):
        model_path = Path(saved_model_name)
        if device != 'cpu':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if training:
            if path is None:
                raise ValueError("Dataset path is None. Provide a valid text file path for NerdLM.")
            path_obj = Path(path)
            if not path_obj.is_absolute():
                path_obj = (PROJECT_ROOT / path_obj).resolve()
            if path_obj.suffix.lower() != '.txt':
                print("Sorry, only txt files are supported.")

            if not model_path.is_absolute():
                model_path = (PROJECT_ROOT / model_path).resolve()

            vocab_size = CustomDataset(str(path_obj)).get_word_map()
            self.model = DeepTransformer(
                d_model=512,
                d_ff=2048,
                num_heads=8,
                num_layers_gru=2,
                vocab_size=vocab_size,
                device=device
            )
            if saved_model and model_path.is_file():
                if model_path.stat().st_size == 0:
                    print(f"Saved model file is empty at '{model_path}'. Initializing a new model.")
                else:
                    try:
                        self.model.load_state_dict(torch.load(str(model_path), map_location=device))
                    except (EOFError, RuntimeError, ValueError) as exc:
                        print(f"Failed to load saved model at '{model_path}': {exc}. Initializing a new model.")
            elif saved_model:
                print(f"Saved model not found at '{model_path}'. Initializing a new model.")

            self.training = TrainingEvaluating(str(path_obj), test_path, device=device)

            self.customer_dataset = CustomDataset(str(path_obj))
            self.word_map = self.customer_dataset.get_word_map()

        if inference:
            self.predictor = Predictor(model_path)


    def train(self, epochs=10, save_model: bool = True, save_path: str ='nerdlm.pt'):
        self.training.train(epochs=epochs)
        if save_model:
            self.training.save_model(save_path)

    def large_train(self, paths: list, epochs=10, save_model: bool = True, save_path: str ='nerdlm.pt'):
        for path in paths:
            self.training = TrainingEvaluating(path, test_path=None)
            self.training.train(epochs=epochs)

        if save_model:
            self.training.save_model(save_path)


    def generate_answer(self, question):
        if not question:
            print("Please enter a question.")

        sequences = self.predictor.predict(question)

        text = ''.join(sequences)

        return text

if __name__ == "__main__":
    bot = NerdLM('app/model/datasets/english_qa/extended_qa_dataset.txt', training=True, inference=False)
    files = glob.glob(os.path.join('datasets/english_qa', '*.txt'))
    # bot.large_train(files, epochs=100)
    bot.train()