from st_pages import Page, add_page_title, show_pages
import streamlit as st
from PIL import Image
import os

show_pages(
    [   
        Page('home.py','INTRO','🏠'),
        Page("book-ost.py", "START", "🎧"),
    ]
)
add_page_title()
st.divider()

st.markdown('근래에 들어 **한국인의 독서량 감소**와 **젊은 층의 문해력 저하**가 사회적 문제로 떠오르고 있습니다. 책이나 신문과 같은 출판물로 정보를 습득했던 과거와 달리, 오늘날 사람들은 책 이외의 수많은 정보 매체와 미디어로부터 정보를 습득할 수 있게 되며 자연스럽게 독서량이 감소해오고 있습니다.')
st.markdown('미디어를 통한 정보 습득과 달리, 독서는 정제되지 않은 정보를 스스로 이해하고 자신의 것으로 습득하는 지적 과정을 거치기 때문에 독서가 문해력과 같은 지적 능력 발달에 매우 중요한 것으로 알려져 있습니다. 따라서 젊은 층의 문해력 저하 문제의 원인이 ‘독서량 감소’에 있다는 의견이 제기되고 있습니다.')
st.markdown('이러한 **한국인의 독서량 감소**와 **젊은 층의 문해력 저하**에 대하여, 저희 팀은 **독서에 대한 흥미를 높이고 독서를 장려할 수 있는 방안을 제시하는 것**이 두 문제의 해결 방안이 될 것이라 생각했습니다.')
st.markdown('')
path = os.path.dirname('/Users/seoeunseo/Desktop/deep.daiv/프로젝트/project_run/이미지')

img = Image.open(path+'/멜로디책방.png')
st.image(img)
st.markdown('')
st.markdown('''영화나 드라마처럼 **책에도 ost가 필요하다는 Jtbc 멜로디책방 프로그램**으로부터 영감을 얻어, **도서 맞춤 음악 추천 시스템**이라는 주제를 선정했습니다. 자신이 읽고 있는 책을 입력하면 책과 잘 어울리는 음악을 추천해줌으로써 **책의 감정과 내용을 음악 함께 더욱 깊이 음미하는 독서 경험을 제공**하고자 합니다. 젊은 층에게 친숙한 음악을 독서와 결합함으로써 독서에 대한 흥미와 즐거움을 더하고, 장기적으로 독서를 장려하는 하나의 문화적 서비스가 될 수 있을 것으로 기대하고 있습니다.''')

st.header('전체적인 프로세스')
st.markdown('--------------------------------------------------------------------------------------')
st.image('프로세스.png')
st.markdown('--------------------------------------------------------------------------------------')

st.markdown('#####  1️⃣ 읽고 있는 도서 입력')
st.markdown('노래를 추천 받고 싶은 도서의 제목 입력')
st.markdown('')
st.markdown('##### 2️⃣ 입력한 도서와 노래 간 유사도 분석')
st.markdown('도서와 노래의 **①감정적 특성** + **②내용 키워드**를 기반으로 유사도를 계산하는 Content-based Filtering (CBF) 방식을 사용')
st.markdown('')
st.markdown('#####  3️⃣ 유사도가 높은 순서대로 추천 노래 플레이리스트 제공')

st.header('**Example**')
with st.expander("날씨가 좋으면 찾아가겠어요"):
    st.markdown(' ')
    st.markdown('감정적 특성 유사도와 내용 유사도에 여러가지 가중치를 부여하여 결과를 개선해 본 결과')
    st.markdown('**- audio feature 감정 : 가사 감정 : 가사 내용 = 0.8 : 0.1 : 0.1 ( audio feature 기반 유사도 중심)**')
    st.markdown('**- audio feature 감정 : 가사 감정 : 가사 내용 = 0 : 0.5 : 0.5 ( 가사 기반 유사도만 사용)**')
    st.markdown('의 비율로 가중치를 부여 했을 때 좋은 추천이 이루어 짐을 확인 할 수 있었습니다.')
    st.image('예시.png')
