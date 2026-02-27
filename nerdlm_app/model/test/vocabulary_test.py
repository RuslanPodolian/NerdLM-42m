from nerdlm_app.model.vocabulary import Vocabulary

class VocabularyReadingTest:
    def __init__(self):
        self.vocabulary = Vocabulary()

    def test_vocabulary_reading(self):
        lines = self.vocabulary.load_lines_of_text_file('datasets/extended_qa_dataset.txt')
        print(lines)

if __name__ == '__main__':
    vocabulary_reading_test = VocabularyReadingTest()
    vocabulary_reading_test.test_vocabulary_reading()