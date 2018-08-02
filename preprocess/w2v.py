import gzip
import gensim
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors

doc = open("data/all.fr",encoding="utf8")

text=[]
for t in doc:
        text.append(t)


model = gensim.models.Word2Vec(
        text,
        size=150,
        window=10,
        min_count=1,
        workers=10)
model.train(text, total_examples=len(text), epochs=10)
model.save("modelfr")
model.wv.save_word2vec_format("embeddingsfr", fvocab="vocabfr", binary=False)
