import gensim.parsing.preprocessing as gsp
from gensim import utils
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))


filters = [
           gsp.strip_tags,
           gsp.strip_punctuation,
           gsp.strip_multiple_whitespaces,
           gsp.strip_numeric,
           gsp.remove_stopwords,
           gsp.strip_short,
           gsp.stem_text
          ]

def clean_text(s):
    s = s.lower()
    s = utils.to_unicode(s)
    for f in filters:
        s = f(s)
    return s



s = 'UserWarning: paramiko missing, opening SSH/SCP/SFTP paths will be disabled.'
s = clean_text(s)
print(s)

c_tokens = word_tokenize(s)
c_filtered_tokens = [lemmatizer.lemmatize(w) for w in c_tokens if not w in stop_words]

print(c_filtered_tokens)


