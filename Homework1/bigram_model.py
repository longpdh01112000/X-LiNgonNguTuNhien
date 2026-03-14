import math
import random
from collections import defaultdict

START = "<s>"
END = "</s>"


class BigramModel:

    def __init__(self):
        self.unigram_counts = defaultdict(int)
        self.bigram_counts = defaultdict(int)
        self.vocab = set()

    def train(self, corpus_file):

        with open(corpus_file, "r", encoding="utf-8") as f:
            sentences = [line.strip().lower() for line in f if line.strip()]

        for sentence in sentences:

            words = sentence.split()
            tokens = [START] + words + [END]

            for w in tokens:
                self.unigram_counts[w] += 1
                self.vocab.add(w)

            for i in range(len(tokens) - 1):
                bigram = (tokens[i], tokens[i + 1])
                self.bigram_counts[bigram] += 1

        print("Số câu:", len(sentences))
        print("Vocabulary size:", len(self.vocab))
        print("Số bigram:", len(self.bigram_counts))

    def bigram_prob(self, w2, w1):

        count_bigram = self.bigram_counts[(w1, w2)]
        count_unigram = self.unigram_counts[w1]

        if count_unigram == 0:
            return 0

        return count_bigram / count_unigram

    def sentence_probability(self, sentence):

        words = sentence.lower().split()
        tokens = [START] + words + [END]

        prob = 1

        print("\nChi tiết bigram:")

        for i in range(len(tokens) - 1):

            w1 = tokens[i]
            w2 = tokens[i + 1]

            p = self.bigram_prob(w2, w1)

            print(f"P({w2}|{w1}) = {p}")

            prob *= p

        return prob

    def generate_sentence(self, max_len=15):

        word = START
        sentence = []

        while word != END and len(sentence) < max_len:

            candidates = []
            probs = []

            for (w1, w2), count in self.bigram_counts.items():
                if w1 == word:
                    candidates.append(w2)
                    probs.append(count)

            if not candidates:
                break

            word = random.choices(candidates, weights=probs)[0]

            if word != END:
                sentence.append(word)

        return " ".join(sentence)


def main():

    print("BIGRAM LANGUAGE MODEL (VIETNAMESE)")

    model = BigramModel()

    model.train("vietnamese_corpus.txt")

    print("\nTop Bigram Counts:")
    for i, ((w1, w2), count) in enumerate(model.bigram_counts.items()):
        if i > 10:
            break
        print(f"{w1} -> {w2} : {count}")

    test_sentence = "Hôm nay trời đẹp lắm"

    print("\nTính xác suất câu:")
    print(test_sentence)

    prob = model.sentence_probability(test_sentence)

    print("\nSentence probability:", prob)

    print("\nSinh câu từ mô hình:")

    for i in range(5):
        print(i + 1, model.generate_sentence())


if __name__ == "__main__":
    main()