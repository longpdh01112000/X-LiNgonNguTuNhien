# ==============================
# 1. IMPORT & DOWNLOAD DATA
# ==============================
import nltk
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import brown
from nltk.tag import UnigramTagger, BigramTagger
from nltk.tag.hmm import HiddenMarkovModelTrainer
from nltk import pos_tag

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

nltk.download('brown')
nltk.download('universal_tagset')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng') 

# ==============================
# 2. LOAD DATA
# ==============================
sentences = brown.tagged_sents(tagset='universal')

train_data = sentences[:4000]
test_data = sentences[4000:5000]


# ==============================
# 3. BUILD MODELS
# ==============================

# 🔹 Unigram
unigram_tagger = UnigramTagger(train_data)

# 🔹 Bigram (fallback = unigram)
bigram_tagger = BigramTagger(train_data, backoff=unigram_tagger)

# 🔹 Perceptron
def perceptron_tag(sentence):
    words = [w for w, t in sentence]
    return pos_tag(words, tagset='universal')

# 🔹 HMM
trainer = HiddenMarkovModelTrainer()
hmm_tagger = trainer.train(train_data)


# ==============================
# 4. PREDICT
# ==============================
y_true = []

y_pred_uni = []
y_pred_bi = []
y_pred_per = []
y_pred_hmm = []

for sent in test_data:
    words = [w for w, t in sent]
    true_tags = [t for w, t in sent]

    # Unigram
    uni = unigram_tagger.tag(words)
    uni_tags = [t if t else 'NOUN' for w, t in uni]

    # Bigram
    bi = bigram_tagger.tag(words)
    bi_tags = [t if t else 'NOUN' for w, t in bi]

    # Perceptron
    per = perceptron_tag(sent)
    per_tags = [t for w, t in per]

    # HMM
    hmm = hmm_tagger.tag(words)
    hmm_tags = [t for w, t in hmm]

    y_true.extend(true_tags)
    y_pred_uni.extend(uni_tags)
    y_pred_bi.extend(bi_tags)
    y_pred_per.extend(per_tags)
    y_pred_hmm.extend(hmm_tags)


# ==============================
# 5. EVALUATE FUNCTION
# ==============================
def evaluate(name, y_true, y_pred):
    p = precision_score(y_true, y_pred, average='macro')
    r = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"\n{name}")
    print("Precision:", round(p, 4))
    print("Recall:", round(r, 4))
    print("F1-score:", round(f1, 4))

    return p, r, f1


# ==============================
# 6. EVALUATION
# ==============================
print("===== RESULT =====")

evaluate("Unigram Tagger", y_true, y_pred_uni)
evaluate("Bigram Tagger", y_true, y_pred_bi)
evaluate("Perceptron Tagger", y_true, y_pred_per)
evaluate("HMM Tagger", y_true, y_pred_hmm)


# ==============================
# 7. CONFUSION MATRIX (BONUS)
# ==============================
labels = list(set(y_true))

cm = confusion_matrix(y_true, y_pred_per, labels=labels)

plt.figure(figsize=(10, 8))
plt.imshow(cm)
plt.title("Confusion Matrix - Perceptron")
plt.colorbar()

plt.xticks(range(len(labels)), labels, rotation=90)
plt.yticks(range(len(labels)), labels)

plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()
plt.show()