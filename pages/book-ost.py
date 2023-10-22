import streamlit as st
import pandas as pd
import warnings
import openpyxl

warnings.filterwarnings('ignore')

import requests
import json

from sklearn.metrics.pairwise import cosine_similarity
import time
from bs4 import BeautifulSoup as bs

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

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

@st.cache_data
def load_data():
    return pd.read_excel('final_data.xlsx',index_col=0)
data = load_data()

@st.cache_data
def load_lyrics():
    return pd.read_excel('lyrics.xlsx',index_col=0)
lyrics = load_lyrics()

@st.cache_data
def load_model():
    return joblib.load('SVM.pkl')
model = load_model()

############################
st.header('1️⃣ 책 제목을 입력해주세요.')

book_title =  st.text_input(label = '예시) 날씨가 좋으면 찾아가겠어요',value="",key='text')

def reset():
    st.session_state.text = ""

reset = st.button('Reset',on_click=reset)
if not book_title:
        con = st.container()
        con.caption('Result')
        con.error('책 제목을 입력해주세요.',icon="⚠️")
        st.stop()


rest_api_key = "41d651c93152d5ec054dc828cacfa671"
url = "https://dapi.kakao.com/v3/search/book"
header = {"authorization": "KakaoAK "+rest_api_key}
querynum = {"query": book_title}

try:
    response = requests.get(url, headers=header, params = querynum)
    content = response.text
    책정보 = json.loads(content)['documents'][0]
except:
    con = st.container()
    con.caption('Result')
    con.error('존재하지 않는 책입니다. 다시 입력해주세요.',icon="🚨")
    st.stop()

book = pd.DataFrame({'title': 책정보['title'],
              'isbn': 책정보['isbn'],
              'authors': 책정보['authors'],
              'publisher': 책정보['publisher']})

target_url = 책정보['url']


response = requests.get(target_url)
soup = bs(response.text, "html.parser")

책소개 = soup.select('#tabContent > div:nth-child(1) > div:nth-child(3) > p')
책속으로 = soup.select('#tabContent > div:nth-child(1) > div:nth-child(6) > p')
서평 = soup.select('#tabContent > div:nth-child(1) > div:nth-child(7) > p')

책소개 = 책소개[0].text
책속으로 = 책속으로[0].text
서평 = 서평[0].text

book['책소개'] = 책소개
book['책속으로'] = 책속으로
book['서평'] = 서평

img= soup.select('#tabContent > div:nth-child(1) > div.info_section.info_intro > div.wrap_thumb > span > img')
img_src = img[0]['src']

col1, col2 = st.columns([1,2])
with col1:
    st.image(img_src,width=150)
with col2:
    title = book['title'][0]
    author = book['authors'][0]
    publisher = book['publisher'][0]
    
    st.caption('제목 : '+ title)
    st.caption('저자 : '+ author)
    st.caption('출산사 : '+publisher)

st.title('')
text = '<'+title +'>에 대한 정보를 모으고 있는 중입니다.'
my_bar = st.progress(0, text=text)
time.sleep(5)
my_bar.progress(5, text='〰️5%〰️')


time.sleep(1)
my_bar.progress(30, text='〰️30%〰️')


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
    
    del tokens

    text_t=list()
    for _ in pos_tokens:
        text_t.append([_[0], get_wordnet_pos(_[1])])
    
    del pos_tokens
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word[0], word[1]) for word in text_t])
    del lemmatizer
    return text

def clean(text):
    text = remove_special_characters(text, remove_digits=True)
    text = delete_stops(text)
    text = tockenize(text)
    return text


translator = Translator()
for col in ['책소개', '책속으로', '서평']:
    name = col+'_trans'
    if book[col].values == '':
        book[name] = ''
        continue
    book[name] = clean(translator.translate(hapus_url(book.loc[0, col])).text)
del stops
del translator

total_text = book.loc[0, '책소개_trans'] + book.loc[0, '책속으로_trans'] + book.loc[0, '서평_trans']
long = book.loc[0, '책소개'] + book.loc[0, '책속으로'] + book.loc[0, '서평']

del book

@st.cache_data
def load_tweet():
    return pd.read_csv('tweet_data_agumentation.csv', index_col = 0)
df = load_tweet()

tfidf_vect_emo = TfidfVectorizer()
tfidf_vect_emo.fit_transform(df["content"])

del df

total_text2 = tfidf_vect_emo.transform(pd.Series(total_text))
model.predict_proba(total_text2)
sentiment = pd.DataFrame(model.predict_proba(total_text2),index=['prob']).T
sentiment['감정'] = ['empty','sadness','enthusiasm','worry','love','fun','hate','happiness','boredom','relief','anger']

del tfidf_vect_emo
del model

my_bar.progress(60, text='〰️60%〰️')

# audio feature랑 text 감정
audio_data = data.iloc[:,-12:-1]
sentiment_prob = sentiment['prob']
sentiment_prob.index = sentiment['감정']
audio_data.columns = ['empty', 'sadness', 'enthusiasm', 'worry', 'love', 'fun', 'hate',
       'happiness', 'boredom', 'relief', 'anger']
audio_data_1 = pd.concat([sentiment_prob,audio_data.T],axis=1).T

col = ['book']+list(data['name'])
cosine_sim_audio = cosine_similarity(audio_data_1)
cosine_sim_audio_df = pd.DataFrame(cosine_sim_audio, index = col, columns=col)
audio_sim = cosine_sim_audio_df['book']

del audio_data
del cosine_sim_audio
del cosine_sim_audio_df

# 가사랑 text
lyrics_data = data.iloc[:,5:-12]
lyrics_data_1 = pd.concat([sentiment_prob,lyrics_data.T],axis=1).T
cosine_sim_lyrics = cosine_similarity(lyrics_data_1)
cosine_sim_lyrics_df = pd.DataFrame(cosine_sim_lyrics, index =col, columns=col)
lyrics_sim = cosine_sim_lyrics_df['book']
del lyrics_data
del lyrics_data_1 
del cosine_sim_lyrics
del cosine_sim_lyrics_df
del sentiment_prob
my_bar.progress(80, text='〰️80%〰️')

# 키워드랑 text
keyword_data = data['key_word']
book_song_cont1 = pd.DataFrame({"text": total_text}, index = range(1))
book_song_cont2 = pd.DataFrame({"text": keyword_data})
keyword_data_1 = pd.concat([book_song_cont1, book_song_cont2], axis=0).reset_index(drop=True)

tfidf_vect_cont = TfidfVectorizer()
tfidf_matrix_cont = tfidf_vect_cont.fit_transform(keyword_data_1['text'])
tfidf_array_cont = tfidf_matrix_cont.toarray()

cosine_sim_keyword = cosine_similarity(tfidf_array_cont)
cosine_sim_keyword_df = pd.DataFrame(cosine_sim_keyword, index = col, columns=col)
keyword_sim = cosine_sim_keyword_df['book']

del total_text
del keyword_data
del book_song_cont1 
del book_song_cont2
del keyword_data_1 
del tfidf_vect_cont
del tfidf_matrix_cont 
del tfidf_array_cont 
del cosine_sim_keyword 
del cosine_sim_keyword_df


my_bar.progress(100, text='100%')

# 전체 유사도 계산
total_sim  = 0.8*audio_sim + 0.1*lyrics_sim + 0.1*keyword_sim

total_sim_df = pd.DataFrame(total_sim[1:])
total_sim_df = total_sim_df.reset_index()
total_sim_df.columns = ['name','book']

top_five = total_sim_df.sort_values(by='book',ascending=False)[:5]
index = total_sim_df.sort_values(by='book',ascending=False)[:5].index.sort_values()

del total_sim
del total_sim_df

artist = data.iloc[index][['url','name','Artist']]
top_five_df = pd.merge(artist,top_five,on='name').sort_values(by='book',ascending=False).drop_duplicates()

del artist 
del top_five 

total_sim  = 0*audio_sim + 0.5*lyrics_sim + 0.5*keyword_sim

total_sim_df_1 = pd.DataFrame(total_sim[1:])
total_sim_df_1 = total_sim_df_1.reset_index()
total_sim_df_1.columns = ['name','book']

top_five_1 = total_sim_df_1.sort_values(by='book',ascending=False)[:5]
index_1 = total_sim_df_1.sort_values(by='book',ascending=False)[:5].index.sort_values()

del total_sim 
del total_sim_df_1

artist = data.iloc[index_1][['url','name','Artist']]
top_five_df_1 = pd.merge(artist,top_five_1,on='name').sort_values(by='book',ascending=False).drop_duplicates()

del artist
del top_five_1
del data

time.sleep(1)
my_bar.empty()


st.caption('책 소개 중....')
st.markdown(long[:300]+'...')

st.markdown('')

lyrics_list = []
for i in top_five_df['url']:
    lyrics_list.append(lyrics[i== lyrics['url']]['lyrics'].values[0])
for i in top_five_df_1['url']:
    lyrics_list.append(lyrics[i== lyrics['url']]['lyrics'].values[0])

lyrics_eng_list = []
for i in top_five_df['url']:
    lyrics_eng_list.append(lyrics[i== lyrics['url']]['lyrics_english'].values[0])
for i in top_five_df_1['url']:
    lyrics_eng_list.append(lyrics[i== lyrics['url']]['lyrics_english'].values[0])

del lyrics


st.header('2️⃣ 결과')
st.subheader('🙂 노래와 분위기가 유사한 노래')
st.caption('AF : 가사 : 키워드 = 0.8 : 0.1 : 0.1')
tab1, tab2, tab3, tab4, tab5= st.tabs(['TOP 1' , 'TOP 2', 'TOP 3', 'TOP 4', 'TOP 5'])
with tab1:
    st.subheader('🥇 TOP 1')
    st.markdown('**제목** : {0}'.format(top_five_df.iloc[0]['name']))
    st.markdown('**가수** : {0} '.format(top_five_df.iloc[0]['Artist']))
    st.markdown('**url** : {0} '.format(top_five_df.iloc[0]['url']))
    st.markdown('**유사도** : {0:.4f}'.format(top_five_df.iloc[0]['book']))
    with st.expander('가사'):
        st.caption('원본 ver')
        st.markdown(lyrics_list[0])
        st.caption('영어 ver')
        st.markdown(lyrics_eng_list[0])
    st.markdown('')
with tab2:
    st.subheader('🥈 TOP 2')
    st.markdown('**제목** : {0}'.format(top_five_df.iloc[1]['name']))
    st.markdown('**가수** : {0} '.format(top_five_df.iloc[1]['Artist']))
    st.markdown('**url** : {0} '.format(top_five_df.iloc[1]['url']))
    st.markdown('**유사도** : {0:.4f}'.format(top_five_df.iloc[1]['book']))
    with st.expander('가사'):
        st.caption('원본 ver')
        st.markdown(lyrics_list[1])
        st.caption('영어 ver')
        st.markdown(lyrics_eng_list[1])
    st.markdown('')
with tab3:
    st.subheader('🥉 TOP 3')
    st.markdown('**제목** : {0}'.format(top_five_df.iloc[2]['name']))
    st.markdown('**가수** : {0} '.format(top_five_df.iloc[2]['Artist']))
    st.markdown('**url** : {0} '.format(top_five_df.iloc[2]['url']))
    st.markdown('**유사도** : {0:.4f}'.format(top_five_df.iloc[2]['book']))
    with st.expander('가사'):
        st.caption('원본 ver')
        st.markdown(lyrics_list[2])
        st.caption('영어 ver')
        st.markdown(lyrics_eng_list[2])
    st.markdown('')
with tab4:
    st.subheader('TOP 4')
    st.markdown('**제목** : {0}'.format(top_five_df.iloc[3]['name']))
    st.markdown('**가수** : {0} '.format(top_five_df.iloc[3]['Artist']))
    st.markdown('**url** : {0} '.format(top_five_df.iloc[3]['url']))
    st.markdown('**유사도** : {0:.4f}'.format(top_five_df.iloc[3]['book']))
    with st.expander('가사'):
        st.caption('원본 ver')
        st.markdown(lyrics_list[3])
        st.caption('영어 ver')
        st.markdown(lyrics_eng_list[3])
    st.markdown('')
with tab5:
    st.subheader('TOP 5')
    st.markdown('**제목** : {0}'.format(top_five_df.iloc[4]['name']))
    st.markdown('**가수** : {0} '.format(top_five_df.iloc[4]['Artist']))
    st.markdown('**url** : {0} '.format(top_five_df.iloc[4]['url']))
    st.markdown('**유사도** : {0:.4f}'.format(top_five_df.iloc[4]['book']))
    with st.expander('가사'):
        st.caption('원본 ver')
        st.markdown(lyrics_list[4])
        st.caption('영어 ver')
        st.markdown(lyrics_eng_list[4])

st.subheader('📖 노래와 내용이 유사한 노래')
st.caption('AF : 가사 : 키워드 = 0 : 0.5 : 0.5')
tab1, tab2, tab3, tab4, tab5= st.tabs(['TOP 1' , 'TOP 2', 'TOP 3', 'TOP 4', 'TOP 5'])
with tab1:
    st.subheader('🥇 TOP 1')
    st.markdown('**제목** : {0}'.format(top_five_df_1.iloc[0]['name']))
    st.markdown('**가수** : {0} '.format(top_five_df_1.iloc[0]['Artist']))
    st.markdown('**url** : {0} '.format(top_five_df_1.iloc[0]['url']))
    st.markdown('**유사도** : {0:.4f}'.format(top_five_df_1.iloc[0]['book']))
    with st.expander('가사'):
        st.caption('원본 ver')
        st.markdown(lyrics_list[5])
        st.caption('영어 ver')
        st.markdown(lyrics_eng_list[5])
    st.markdown('')
with tab2:
    st.subheader('🥈 TOP 2')
    st.markdown('**제목** : {0}'.format(top_five_df_1.iloc[1]['name']))
    st.markdown('**가수** : {0} '.format(top_five_df_1.iloc[1]['Artist']))
    st.markdown('**url** : {0} '.format(top_five_df_1.iloc[1]['url']))
    st.markdown('**유사도** : {0:.4f}'.format(top_five_df_1.iloc[1]['book']))
    with st.expander('가사'):
        st.caption('원본 ver')
        st.markdown(lyrics_list[6])
        st.caption('영어 ver')
        st.markdown(lyrics_eng_list[6])
    st.markdown('')
with tab3:
    st.subheader('🥉 TOP 3')
    st.markdown('**제목** : {0}'.format(top_five_df_1.iloc[2]['name']))
    st.markdown('**가수** : {0} '.format(top_five_df_1.iloc[2]['Artist']))
    st.markdown('**url** : {0} '.format(top_five_df_1.iloc[2]['url']))
    st.markdown('**유사도** : {0:.4f}'.format(top_five_df_1.iloc[2]['book']))
    with st.expander('가사'):
        st.caption('원본 ver')
        st.markdown(lyrics_list[7])
        st.caption('영어 ver')
        st.markdown(lyrics_eng_list[7])
    st.markdown('')
with tab4:
    st.subheader('TOP 4')
    st.markdown('**제목** : {0}'.format(top_five_df_1.iloc[3]['name']))
    st.markdown('**가수** : {0} '.format(top_five_df_1.iloc[3]['Artist']))
    st.markdown('**url** : {0} '.format(top_five_df_1.iloc[3]['url']))
    st.markdown('**유사도** : {0:.4f}'.format(top_five_df_1.iloc[3]['book']))
    with st.expander('가사'):
        st.caption('원본 ver')
        st.markdown(lyrics_list[8])
        st.caption('영어 ver')
        st.markdown(lyrics_eng_list[8])
    st.markdown('')
with tab5:
    st.subheader('TOP 5')
    st.markdown('**제목** : {0}'.format(top_five_df_1.iloc[4]['name']))
    st.markdown('**가수** : {0} '.format(top_five_df_1.iloc[4]['Artist']))
    st.markdown('**url** : {0} '.format(top_five_df_1.iloc[4]['url']))
    st.markdown('**유사도** : {0:.4f}'.format(top_five_df_1.iloc[4]['book']))
    with st.expander('가사'):
        st.caption('원본 ver')
        st.markdown(lyrics_list[9])
        st.caption('영어 ver')
        st.markdown(lyrics_eng_list[9])
