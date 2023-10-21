import streamlit as st
import pandas as pd
import warnings
import openpyxl
from webdriver_manager.chrome import ChromeDriverManager

warnings.filterwarnings('ignore')

import requests
import json

from sklearn.metrics.pairwise import cosine_similarity

from selenium import webdriver
from selenium.webdriver.common.by import By

import time

from googletrans import Translator

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

data = pd.read_excel('final_data.xlsx',index_col=0)

st.subheader('📗 추천 도서 리스트')
with st.expander('AF : 가사 : 키워드 = 0.8 : 0.1 : 0.1 인 경우'):
    col1,col2,col3= st.columns([1,1,1])
    with col1:
        st.markdown('**1. 참을 수 없는 존재의 가벼움**')
        st.image('https://velog.velcdn.com/images/jeo0534/post/71f6da54-1ac8-4581-8f77-f55ed5c56dbc/image.png')
        st.caption('사랑은 은유로 시작된다. 달리 말하자면, 한 여자가 언어를 통해 우리의 시적 기억에 아로새겨지는 순간, 사랑은 시작되는 것이다.')
        st.caption('그들은 서로 사랑했는데도 상대방에게 하나의 지옥을 선사했다.')
    with col2:
        st.markdown('**2. 이성과 감성**')
        st.image('https://velog.velcdn.com/images/jeo0534/post/74f8850c-4d18-42f5-ac29-7e3a1f4f68f4/image.png')
        st.caption('"이성"과 "감성"이라는 두 가지 인간성을 연애와 결혼이라는 보편적 주제를 통한 고찰')
    with col3:
        st.markdown('**3. 지금, 만나러 갑니다**')
        st.image('https://velog.velcdn.com/images/jeo0534/post/fca0c4eb-51bb-4b3c-93d0-866ed37f60fc/image.png')
        st.caption('당신에겐 있나요? 기적같은 단 한사람')
        st.caption('그 사람을 다시 한 번 만날 수 있다면.')
        st.caption('')
        st.caption('더 이상 볼 수 없게 된 그리운 사람과의 기적 같은 재회를 그린다. 1년 전 세상을 떠난 아내 미오를 그리워하며 하루하루를 보내는 다쿠미는 비 오는 날 아들 유지와 함께 찾은 숲속에서 놀랍게도 죽은 미오와 재회한다. 이야기는 누구보다 차근차근 마음을 쌓아가며 느리게 사랑해온 두 사람의 과거로 거슬러 올라간다.')
    
    col4,col5,col6= st.columns([1,1,1])
    with col4:
        st.markdown('**4. 모순**')
        st.image('https://velog.velcdn.com/images/jeo0534/post/66e26a96-c2d5-4c15-a931-66093fb0798e/image.png')
        st.caption('인생은 탐구하면서 살아가는 것이 아니라, 살아가면서 탐구하는 것이다. 실수는 되풀이된다. 그것이 인생이다…….')
        st.caption('바로 그 이유 때문에 사랑을 시작했고, 바로 그 이유 때문에 미워하게 된다는, 인간이란 존재의 한없는 모순......')
    with col5:
        st.markdown('**5. 사랑의 파괴**')
        st.image('https://velog.velcdn.com/images/jeo0534/post/0192b5f2-7218-42b7-a5dc-7fa66f654f1c/image.png')
        st.caption('엘레나는 자신을 위해서 내가 나 자신을 파괴하기를 원하고 있었다.')
        st.caption('사랑하는 만큼 사랑받고자 하는 욕망, 순진하기에 더욱더 잔혹한 유년의 사랑')
    with col6:
        st.markdown('**6. 제인에어**')
        st.image('https://velog.velcdn.com/images/jeo0534/post/e2c36e8d-7446-44af-a387-3f43e968d713/image.png')
        st.caption('순응하고 인내하며 봉사하는 여성이 이상적으로 여겨지던 빅토리아 시대에, 현실적인 조건이나 개인적 자질에서 이와 동떨어진 여성인 제인의 성장을 통해 당대 여성의 삶 전반, 즉 여성의 교육, 고용, 사랑, 결혼에 대한 의문')
        
    col7,col8,col9= st.columns([1,1,1])
    with col7:
        st.markdown('**7. 무의미의 축제**')
        st.image('https://velog.velcdn.com/images/jeo0534/post/8e967400-dc99-4d63-8ed1-94ab93396b0e/image.png')
        st.caption('보잘것없는 것을 사랑해야 해요,사랑하는 법을 배워야 해요.')
        st.caption('농담과 거짓말, 의미와 무의미, 일상과 축제의 경계에서삶과 인간의 본질을 바라보는 시선')
    with col8:
        st.markdown('**8. 80일간의 세계일주**')
        st.image('https://velog.velcdn.com/images/jeo0534/post/8875c9d4-d568-4242-a478-4358e97411df/image.png')
        st.caption('2만 파운드를 걸고 80일 동안의 세계 일주에 나선 영국 신사 필리어스 포그. ')
        st.caption('그는 기계처럼 정확하고 냉정한 영국 신사다. 한 치의 오차도 없이 여행을 계획하는 주인공을 통해 쥘 베른은 치밀하고 과학적이며 이성적인 인간과, 인간에 대한 신뢰와 애정 그리고 세계에 대한 긍정으로 차 있는 인간상을 그려 낸다.')
    with col9:
        st.markdown('**9. 몬테크리스토 백작**')
        st.image('https://velog.velcdn.com/images/jeo0534/post/aef3eb6b-e3d9-4745-8b7b-07e592e4637b/image.png')
        st.caption('모든 악에는 두 개의 약이 있다. 시간과 침묵이 그것이다')
        st.caption('인간사에서 가장 흥겨운 이야기는 불행을 딛고 행복을 되찾는 이야기가 아닐까?')
        st.caption('모략과 함정에 빠지지만, 부와 명예를 회복하여 화려하게 복수한다는 이야기에 사람들은 쉽게 열광한다.')
        st.caption('<몬테크리스토 백작>이 대표적인 경우. 배신, 억울한 감금, 복수 이 3요소는 시대를 불문하고 독자들을 매료시켰다.')

    col10,col11,col12 = st.columns([1,1,1])
    with col10:
        st.markdown('**10. 페드르와 이폴리트**')
        st.image('https://velog.velcdn.com/images/jeo0534/post/fc8c56f5-5661-427b-bb39-70c6e6169fe4/image.png')
        st.caption('인간은 진정 자신을 옥죄는 정념으로부터 스스로를 구할 의지도, 능력도 없는 존재인가.')
        st.caption('에우리피데스의 「히폴리토스」를 바탕으로 정념이 지닌 파괴적 본성,통제할 수 없는 정념에 빠진 한 인간이 보여 주는 감정의 격정을 파고든 라신 비극의 정수.')
    with col11:
        st.markdown('**11. 결혼ㆍ여름**')
        st.image('https://velog.velcdn.com/images/jeo0534/post/81bbcf18-81e5-4952-853b-927bb8c2223d/image.png')
        st.caption('깊이 사랑하는 여인의 매력을 항목별로 조목조목 읊을 수 있겠는가?그럴 수 없다, 그냥 전체를 사랑하는 것이다.')
        st.caption('카뮈 사상의 핵심인 ‘부조리’와 ‘반항’의 출발 및 완성 과정이 육성으로 들리는 듯한 자전적 기록')

with st.expander('가사 중심인 경우'):
    col1,col2,col3= st.columns([1,1,1])
    with col1:
        st.markdown('**1. 참을 수 없는 존재의 가벼움**')
        st.image('https://velog.velcdn.com/images/jeo0534/post/71f6da54-1ac8-4581-8f77-f55ed5c56dbc/image.png')
        st.caption('사랑은 은유로 시작된다. 달리 말하자면, 한 여자가 언어를 통해 우리의 시적 기억에 아로새겨지는 순간, 사랑은 시작되는 것이다.')
        st.caption('그들은 서로 사랑했는데도 상대방에게 하나의 지옥을 선사했다.')
    with col2:
        st.markdown('**2. 어린왕자**')
        st.image('https://velog.velcdn.com/images/jeo0534/post/93a93b74-d728-4727-81b5-7af32114a2aa/image.png')
        st.caption('네가 오후 4시에 온다면 난 3시부터 설렐 거야. 4시가 가까워질수록 점점 더 행복해지겠지. 4시가 되면 난 가슴이 두근거려서 안절부절못하고 걱정을 할 거야. 행복의 대가를 알게 되겠지! 하지만 네가 아무 때나 온다면 언제부터 마음의 준비를 해야 할지 도무지 알 수 없잖아.')
        st.caption('순수성을 허락하지 않는 세상에서 끊임없이 방황하고 고뇌한 생텍쥐페리. 그는 세상을 바꿀 수는 없지만 희망을 그리고 싶었고, 자신이 동경하고 희망하는 삶을 ‘어린 왕자’로 형상화했다.')
    with col3:
        st.markdown('**3. 몬테크리스토 백작**')
        st.image('https://velog.velcdn.com/images/jeo0534/post/aef3eb6b-e3d9-4745-8b7b-07e592e4637b/image.png')
        st.caption('모든 악에는 두 개의 약이 있다. 시간과 침묵이 그것이다')
        st.caption('인간사에서 가장 흥겨운 이야기는 불행을 딛고 행복을 되찾는 이야기가 아닐까?')
        st.caption('모략과 함정에 빠지지만, 부와 명예를 회복하여 화려하게 복수한다는 이야기에 사람들은 쉽게 열광한다.')
        st.caption('<몬테크리스토 백작>이 대표적인 경우. 배신, 억울한 감금, 복수 이 3요소는 시대를 불문하고 독자들을 매료시켰다.')
    
    col4,col5,col6= st.columns([1,1,1])
    with col4:
        st.markdown('**4. 로미오와 줄리엣**')
        st.image('https://velog.velcdn.com/images/jeo0534/post/6edfa385-5695-4ba8-a14e-45ce4871f5ca/image.png')
        st.caption('오, 둥근 궤도 안에서 한 달 내내 변하는지조 없는 달에게 맹세하진 마세요')
        st.caption('다쳐 본 적 없는 자가 흉터를 비웃는 법…')
        st.caption('달빛 아래 주고받은 첫 키스와 사랑의 맹세,살아 있는 죽음을 통해 도달하는 죽음을 넘어서는 사랑!셰익스피어가 빚어낸 순수한 열정의 비극, 그 사랑의 모순어법')
    with col5:
        st.markdown('**5. 아주 편안한 죽음**')
        st.image('https://velog.velcdn.com/images/jeo0534/post/86c06fae-d47c-462f-b8bf-2fbfadcd9f33/image.png')
        st.caption('엄마는 유년 시절 내내 규범과 금기라는 갑옷을 두른 채 몸과 마음, 정신을 억압당했다. 그리고 스스로를 끈으로 옭아매도록 교육받았다. 그런 엄마의 내면에는끓어오르는 피와 불같은 정열을 지닌 한 여인이 살아 숨 쉬고 있었다. 그러나 그 여인은 뒤틀리고 훼손된 끝에 자기 자신에게조차 낯선 존재가 되어 버린 모습이었다.')
        st.caption('주체성을 포기하며 타자로 살도록 강요받아 온 한 인간의 생애, 나아가 당대 여성 전체의 모습. ')
        st.caption('냉대하며 외면했던 세계를 새롭게 인식하며 자기 정체성의 일부로 받아들이는 과정이며, 그와 동시에 남과 여, 육체와 정신, 삶과 죽음 등 구별 짓기로 가득했던 인간 내면의 경계를 허무는 작품.')
    with col6:
        st.markdown('**6. 무의미의 축제**')
        st.image('https://velog.velcdn.com/images/jeo0534/post/8e967400-dc99-4d63-8ed1-94ab93396b0e/image.png')
        st.caption('보잘것없는 것을 사랑해야 해요,사랑하는 법을 배워야 해요.')
        st.caption('농담과 거짓말, 의미와 무의미, 일상과 축제의 경계에서삶과 인간의 본질을 바라보는 시선')
    col7,col8,col9= st.columns([1,1,1])
    with col7:
        st.markdown('**7. 잘못 걸려온 전화**')
        st.image('https://velog.velcdn.com/images/jeo0534/post/9fa4d386-e2fa-435d-91bc-7841c5ddd96e/image.png')
        st.caption('그런 식으로 세월은 흘러갈 것이다. 그리고 악몽 같던 내 인생의 장면들이 눈에 선할 것이다. 그러나 나는 이제 그것들로 인해 아파하지 않을 것이다.')
        st.caption('죽음, 사랑, 그리고 상실"아고타 크리스토프의 작품 중 가장 낯설고 비밀스러운 악몽과 우화" - 르 몽드(Le Monde)')
    with col8:
        st.markdown('**8. 파우스트**')
        st.image('https://velog.velcdn.com/images/jeo0534/post/9bc7048f-9e21-43cc-8db8-a71bb1bc8dfa/image.png')
        st.caption('내가 너의 노예가 되어 이 세상 모든 영화를 체험하게 해주는 대신，네가 어느 한순간 `멈추어라．너는 너무도 아름답다’라며 휴식을 원하면 그때부터 너의 영혼은 영원히 나의 것이다.')
        st.caption('지식과 학문에 절망한 노학자 파우스트 박사의 미망(迷妄)과 구원의 장구한 노정을 그린다. 악마 메피스토펠레스의 유혹에 빠져 현세의 쾌락을 쫓으며 방황하던 파우스트는 마침내 잘못을 깨닫고 천상의 구원을 받는다.')
    with col9:
        st.markdown('**9. 어떻게든 이별**')
        st.image('https://velog.velcdn.com/images/jeo0534/post/10ca695a-924b-44be-a573-1227e1525510/image.png')
        st.caption('이 계절은 조금 가벼운 절망을 앓기에 얼마나 찬란한가')
        st.caption('사랑, 결국에는 이별, 끝내 불가피한 고독지극한 상처 안에 웃음을 품은 쓸쓸한 통찰')

st.divider()

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


# 옵션 생성
options = webdriver.ChromeOptions()
# 창 숨기는 옵션 추가
options.add_argument("headless")

#driver = webdriver.Chrome(executable_path=ChromeDriverManager().install(), options=options)
driver = webdriver.Chrome('/Users/seoeunseo/Desktop/deep.daiv/프로젝트/project_run/chromedriver.exe', options=options)
driver.get(target_url)

img= driver.find_element(By.XPATH, '//*[@id="tabContent"]/div[1]/div[1]/div[1]/span/img')
img_src = img.get_attribute('src')

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

try :
    botton = driver.find_element(By.XPATH, '//*[@id="tabContent"]/div[1]/div[2]/div[3]/a')
    botton.click()
except :
    pass
책소개 = driver.find_element(By.XPATH, '//*[@id="tabContent"]/div[1]/div[2]/p')

time.sleep(3)
my_bar.progress(10, text='〰️10%〰️')
try :
    botton = driver.find_element(By.XPATH, '//*[@id="tabContent"]/div[1]/div[5]/div[3]/a')
    botton.click()
except :
    pass
책속으로 = driver.find_element(By.XPATH, '//*[@id="tabContent"]/div[1]/div[5]/p')

time.sleep(3)
my_bar.progress(20, text='〰️20%〰️')
try :
    botton = driver.find_element(By.XPATH, '//*[@id="tabContent"]/div[1]/div[6]/div[3]/a')
    botton.click()
except :
    pass
서평 = driver.find_element(By.XPATH, '//*[@id="tabContent"]/div[1]/div[6]/p')


book['책소개'] = 책소개.text
book['책속으로'] = 책속으로.text
book['서평'] = 서평.text
driver.close()

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


translator = Translator()
for col in ['책소개', '책속으로', '서평']:
    name = col+'_trans'
    if book[col].values == '':
        book[name] = ''
        continue
    book[name] = clean(translator.translate(hapus_url(book.loc[0, col])).text)

total_text = book.loc[0, '책소개_trans'] + book.loc[0, '책속으로_trans'] + book.loc[0, '서평_trans']



df = pd.read_csv('tweet_data_agumentation.csv', index_col = 0)

tfidf_vect_emo = TfidfVectorizer()
tfidf_vect_emo.fit_transform(df["content"])

model = joblib.load('SVM.pkl')
total_text2 = tfidf_vect_emo.transform(pd.Series(total_text))
model.predict_proba(total_text2)
sentiment = pd.DataFrame(model.predict_proba(total_text2),index=['prob']).T
sentiment['감정'] = ['empty','sadness','enthusiasm','worry','love','fun','hate','happiness','boredom','relief','anger']
sentiment2 = sentiment.sort_values(by='prob',ascending=False)

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

# 가사랑 text
lyrics_data = data.iloc[:,5:-12]
lyrics_data_1 = pd.concat([sentiment_prob,lyrics_data.T],axis=1).T
cosine_sim_lyrics = cosine_similarity(lyrics_data_1)
cosine_sim_lyrics_df = pd.DataFrame(cosine_sim_lyrics, index =col, columns=col)
lyrics_sim = cosine_sim_lyrics_df['book']
my_bar.progress(80, text='〰️80%〰️')

# 키워드랑 text
keyword_data = data['key_word']
book_song_cont1 = pd.DataFrame({"text": total_text}, index = range(1))
book_song_cont2 = pd.DataFrame({"text": keyword_data})
keyword_data_1 = pd.concat([book_song_cont1, book_song_cont2], axis=0).reset_index(drop=True)

tfidf_vect_cont = TfidfVectorizer()
tfidf_matrix_cont = tfidf_vect_cont.fit_transform(keyword_data_1['text'])
tfidf_array_cont = tfidf_matrix_cont.toarray()
tfidf_df_cont = pd.DataFrame(tfidf_array_cont, columns=tfidf_vect_cont.get_feature_names_out())

cosine_sim_keyword = cosine_similarity(tfidf_array_cont)
cosine_sim_keyword_df = pd.DataFrame(cosine_sim_keyword, index = col, columns=col)
keyword_sim = cosine_sim_keyword_df['book']

# 전체 유사도 계산
total_sim  = 0.8*audio_sim + 0.1*lyrics_sim + 0.1*keyword_sim

recommend_song = total_sim.sort_values(ascending=False)[1:6].index
total_sim_df = pd.DataFrame(total_sim[1:])
total_sim_df = total_sim_df.reset_index()
total_sim_df.columns = ['name','book']

top_five = total_sim_df.sort_values(by='book',ascending=False)[:5]
index = total_sim_df.sort_values(by='book',ascending=False)[:5].index.sort_values()
artist = data.iloc[index][['url','name','Artist']]
top_five_df = pd.merge(artist,top_five,on='name').sort_values(by='book',ascending=False).drop_duplicates()

total_sim  = 0*audio_sim + 0.5*lyrics_sim + 0.5*keyword_sim

recommend_song = total_sim.sort_values(ascending=False)[1:6].index
total_sim_df_1 = pd.DataFrame(total_sim[1:])
total_sim_df_1 = total_sim_df_1.reset_index()
total_sim_df_1.columns = ['name','book']

top_five_1 = total_sim_df_1.sort_values(by='book',ascending=False)[:5]
index_1 = total_sim_df_1.sort_values(by='book',ascending=False)[:5].index.sort_values()

artist = data.iloc[index_1][['url','name','Artist']]
top_five_df_1 = pd.merge(artist,top_five_1,on='name').sort_values(by='book',ascending=False).drop_duplicates()


my_bar.progress(100, text='100%')
time.sleep(1)
my_bar.empty()


st.caption('책 소개 중....')
long = book.loc[0, '책소개'] + book.loc[0, '책속으로'] + book.loc[0, '서평']
st.markdown(long[:300]+'...')

st.markdown('')

lyrics = pd.read_excel('lyrics.xlsx',index_col=0)
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
