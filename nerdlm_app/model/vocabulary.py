import json
import os

class Vocabulary:
    def __init__(self):
        self.df = 0
        self.word_map = {}
        self.word_map['<pad>'] = 0
        self.word_map['<start>'] = 1
        self.word_map['<end>'] = 2
        self.word_map['<unk>'] = 3
        self.word_map["word_map_coef"] = self.word_map['<unk>'] + 1

        vocabulary_path = os.path.join(os.path.dirname(__file__), 'datasets/vocabulary.json')

        self.save_json_word_map = lambda: self.save_json_dump(vocabulary_path)
        self.load_json_word_map = lambda: self.load_json(os.path.join(vocabulary_path))

        try:
            self.word_map = self.load_json_word_map()
            self.word_map["word_map_coef"] = list(self.word_map.values())[-1] + 1
            print("Vocabulary loaded successfully.")
        except Exception as e:
            print(e)
            self.word_map["word_map_coef"] = self.word_map['<unk>'] + 1
            print("No vocabulary file found or error loading it. Initializing existing base one.")

        self.idx_to_word = {idx: token for token, idx in self.word_map.items()}

        self.punctuation_str = '!@#$%^&*()_+-=[]{}|\\:;”“‘’<>,.?/~` '
        self.max_length = 256



    def add_word_per_line(self, lines, split_factor=None):
        if split_factor is None:
            split_factor = self.punctuation_str

        self.word_map = self.load_json_word_map()
        self.word_map['word_map_coef'] = self.word_map['word_map_coef']

        # Add all punctuation characters to vocabulary
        for punct in split_factor:
            if punct not in self.word_map and punct.strip():
                self.word_map[punct] = self.word_map["word_map_coef"]
                self.word_map['word_map_coef'] += 1

        for line in lines:
            # Split by all punctuation characters
            words = [line]
            for punct in split_factor:
                new_words = []
                for w in words:
                    new_words.extend(w.split(punct))
                words = new_words

            # Add non-empty words to vocabulary
            for word in words:
                word = word.strip()
                if word and word not in self.word_map:
                    self.word_map[word] = self.word_map["word_map_coef"]
                    self.word_map['word_map_coef'] += 1
        self.save_json_word_map()

    def add_word(self, word):
        self.word_map = self.load_json_word_map()
        self.word_map['word_map_coef'] = self.word_map['word_map_coef']
        if word not in self.word_map:
            self.word_map[word] = self.word_map["word_map_coef"]
            self.word_map["word_map_coef"] += 1
        self.save_json_word_map()

    def remove_word(self, word):
        self.word_map = self.load_json_word_map()
        self.word_map['word_map_coef'] = self.word_map['word_map_coef']
        if word in self.word_map:
            del self.word_map[word]
            self.word_map["word_map_coef"] -= 1
        self.save_json_word_map()

    def save_json_dump(self, path, logging=False):
        existing = {}
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    existing = json.load(f)
            except (json.decoder.JSONDecodeError, OSError):
                existing = {}

        existing.update(self.word_map)

        tmp_path = path + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(existing, f)
        os.replace(tmp_path, path)

        if logging:
            print(f"Vocabulary saved to {path}")


    def load_json(self, path):
        if not os.path.exists(path):
            return dict(self.word_map)

        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"Corrupted vocabulary file at '{path}'. Rebuilding from scratch.")
            try:
                os.remove(path)
            except OSError:
                pass
            return dict(self.word_map)

        if 'word_map_coef' not in data:
            max_id = max(v for v in data.values() if isinstance(v, int))
            data['word_map_coef'] = max_id + 1

        return data

    def load_lines_of_text_file(self, path):
        if path is None:
            raise ValueError("Dataset path is None. Provide a valid text file path.")

        lines = []
        with open(path, 'r') as f:
            for raw in f:
                line = raw.strip()
                if line:
                    lines.append(line)

        return lines
