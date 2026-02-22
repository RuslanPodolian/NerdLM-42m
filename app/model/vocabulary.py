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
        self.word_map_coef = self.word_map['<unk>'] + 1
        self.idx_to_word = {idx: token for token, idx in self.word_map.items()}

        self.save_json_dump(os.path.join(os.path.dirname(__file__), 'datasets/vocabulary.json'))

        try:
            self.word_map = self.load_json(os.path.join(os.path.dirname(__file__), 'datasets/vocabulary.json'))
        except Exception as e:
            print(e)
            print("No vocabulary file found or error loading it. Initializing existing base one.")



        self.punctuation_str = '!@#$%^&*()_+-=[]{}|\\:;”“‘’<>,.?/~`'
        self.max_length = 256



    def add_word_per_line(self, lines, split_factor=' '):
        for line in lines:
            for word in line.split(split_factor):
                if word not in self.punctuation_str and word not in self.word_map:
                    self.word_map[word] = self.word_map_coef
                    self.word_map_coef += 1

    def add_word(self, word):
        if word not in self.word_map:
            self.word_map[word] = self.word_map_coef
            self.word_map_coef += 1

    def remove_word(self, word):
        if word in self.word_map:
            del self.word_map[word]
            self.word_map_coef -= 1

    def save_json_dump(self, path):
        # with open(path, 'w') as f:
        #     json.dump(self.word_map, f)
        if os.path.exists(path):
            with open(path, 'r') as f:
                existing = json.load(f)
            existing.update(self.word_map)
        else:
            print("No file was found! Import loaded word_map")
            existing = self.word_map

        with open(path, 'w') as f:
            json.dump(existing, f)
        print(f"Vocabulary saved to {path}")


    def load_json(self, path):
        with open(path, 'r') as f:
            data = json.load(f)

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
