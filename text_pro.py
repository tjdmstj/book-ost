from googletrans import Translator

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

#영어 불용어 사전
stops = set(stopwords.words('english'))

def hapus_url(text):
    mention_pattern = r'@[\w]+'
    cleaned_text = re.sub(mention_pattern, '', text)
    return re.sub(r'http\S+','', cleaned_text)

#특수문자 제거
#영어 대소문자, 숫자, 공백문자(스페이스, 탭, 줄바꿈 등) 아닌 문자들 제거
def remove_special_characters(text, remove_digits=True):
    text=re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


#불용어 제거
def delete_stops(text):
    text = text.lower().split()
    text = ' '.join([word for word in text if word not in stops])
    return text
   
    
#품사 tag 매칭용 함수
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    

#품사 태깅 + 표제어 추출
def tockenize(text):
    tokens=word_tokenize(text)
    pos_tokens=nltk.pos_tag(tokens)
    
    text_t=list()
    for _ in pos_tokens:
        text_t.append([_[0], get_wordnet_pos(_[1])])
    
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word[0], word[1]) for word in text_t])
    return text



def clean(text):
    text = remove_special_characters(text, remove_digits=True)
    text = delete_stops(text)
    text = tockenize(text)
    return text