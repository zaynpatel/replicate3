import gensim
import pandas as pd

# step 1 of data set is to read and explore. commands pd.read_json allow me to tell juypter to read file and hit shift-tab to open file.

# df.head() - I get to see first 5 rows of data. 

df = pd.read_json("reviews_Sports_and_Outdoors_5.json", lines=True)
df.head()

df.shape

# simple preprocessing and tokenization:

# preprocessing = cleaning the data, tokenizing, removing stopwords, pos tagging

df.reviewText[0]

review_Text = gensim.utils.simple_preprocess('This came in on time and I am veru happy with it, I haved used it already and it makes taking out the pins in my glock 32 very easy'
)

review_text = df.reviewText.apply(gensim.utils.simple_preprocess)
review_text

model = gensim.models.Word2Vec(
    window=5,
    min_count=4,
    workers=5
)

model.build_vocab(review_text,progress_per=1000)

model.epochs

model.train(review_text, total_examples=model.corpus_count, epochs=model.epochs)

model.wv.most_similar("water")

model.wv.similarity(w1="water", w2="liquid")

model.wv.most_similar("awful")

model.wv.similarity(w1='good', w2='great')

model.wv.similarity(w1='slow', w2='steady')
