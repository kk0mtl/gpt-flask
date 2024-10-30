from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from pymongo import MongoClient
from pdfkit.configuration import Configuration
import datetime
import pdfkit
import requests
import json
import sys
from dotenv import load_dotenv
import os
import openai
import re
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
import numpy as np
from langchain.docstore.document import Document as LangChainDocument
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.enums import TA_JUSTIFY


app = Flask(__name__)
CORS(app)

load_dotenv()

# MongoDB 연결
MONGODB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGODB_URI)
db = client['test']
collection = db['datas']
collection_s = db['summary']

# 설정
openai.api_key = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

# GPT 번역
@app.route('/translate', methods=['POST'])
def translate_text():
    try:
        data = request.json
        text = data.get('text', '')

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "### instruction ###\n문장을 영어로 번역해줘"},
                {"role": "user", "content": text},
            ]
        )
        print(response)
        translated_text = response.choices[0].message.content.strip()
        return jsonify({'translated_text': translated_text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def format_time_range(start_time, end_time):
    start_str = start_time.strftime('%H시 %M분')
    end_str = end_time.strftime('%H시 %M분')
    return f"{start_str} ~ {end_str}"

# 요약
@app.route('/summarize', methods=['POST'])
def summarize_script():
    # MongoDB에서 모든 문서 가져오기
    documents = collection.find()

    first_document = collection.find_one()
    if first_document:
        first_timestamp = first_document.get('timestamp')
    else:
        first_timestamp = None

    # 사용자 이름과 메시지를 저장할 리스트 생성
    script = []

    # 각 문서에서 사용자 이름과 메시지 추출하여 형식에 맞게 저장
    for document in documents:
        username = document.get('userName')
        sttmsg = document.get('sttMsg')
        script.append({"username": username, "sttmsg": sttmsg})

    # 스크립트를 1000자씩 분리
    parts = [script[i:i+1000] for i in range(0, len(script), 1000)]
    summaries = []

    for part in parts:
        # 파트를 문자열로 변환
        part_content = "\n".join([f"{item['username']}: {item['sttmsg']}" for item in part])

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": part_content},
                {"role": "assistant",
                 "content": f"""
                        ### instruction ###
                        아래 텍스트를 '주요 주제', '다음 할 일', '요약', '키워드'로 정리해줘.
                        4개의 큰 주제 앞에는 '•' 를 붙여서 구분해줘.
                        요약 할 때는 개조식으로 해야 돼.
                        """
                }
            ]
        )

        # 결과 저장
        summary = response.choices[0].message.content
        summaries.append(summary)

    # 결합
    full_summary = ' '.join(summaries)

    # 전체 주제
    topic_response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_summary},
            {"role": "assistant",
             "content": "### instruction ###\n\
                        회의 내용을 관통하는 하나의 주제를 한 문장"}
        ]
    )

    topic = topic_response.choices[0].message.content

    # 현재 날짜 및 시간
    current_date = datetime.datetime.now().strftime('%Y년 %m월 %d일')

    # 결과 summary에 저장
    summary_data = {
        "full_summary": full_summary,
        "current_date": current_date,
        "topic_response": topic,
    }

    collection_s.insert_one(summary_data)

    return jsonify({
         "full_summary": full_summary,
         "current_date": current_date,
         "topic_response": topic,
    })
    
# ==== 회의록 만듦 ==== #
@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    data = request.json
    current_user = data.get('currentUser')
    print("current_user: ", current_user)
    
    latest_summary = collection_s.find_one(sort=[('_id', -1)])  # 최신 문서 기준으로 정렬하여 하나 가져옴
        
    if latest_summary:
        full_summary=latest_summary.get('full_summary')
        current_date = latest_summary.get('current_date')
        main_topic = latest_summary.get('topic_response')
        
        parsed_summary = parse_full_summary(full_summary)
    
        if parsed_summary:
            topic = parsed_summary["topic"]
            todo = parsed_summary["todo"]
            content = parsed_summary["content"]
            
            create_meeting_report(topic, todo, content, current_date, main_topic, current_user)
    
            return send_file("meeting_report.pdf", as_attachment=True)
        
        else:
            return jsonify({"error": "Failed to parse full summary"}), 500
    
    else:
        return jsonify({"error": "No summary found in the database"}), 500

# full_summary parsing
def parse_full_summary(full_summary):
    pattern = r"• 주요 주제\n(.*?)\n• 다음 할 일\n(.*?)\n• 요약\n(.*?)• 키워드\n(.*)"
    match = re.search(pattern, full_summary, re.DOTALL)
    
    if match:
        topic = match.group(1).strip()
        todo = match.group(2).strip()
        content = match.group(3).strip()
        keyword = match.group(4).strip()
        
        return{
            "topic" : topic,
            "todo" : todo,
            "content" : content,
            "keyword": keyword
        }
    else:
        return None
        
# 글꼴 등록
font_path = os.path.join(os.path.dirname(__file__), 'fonts', 'NanumGothic.ttf')
pdfmetrics.registerFont(TTFont('NanumGothic', font_path))

# 스타일 정의
styles = getSampleStyleSheet()
paragraph_style = ParagraphStyle(
    'Normal',
    fontSize= 12,
    leading=14,
    alignment= TA_JUSTIFY,
    wordWrap= 'DJK',
    fontName='NanumGothic'
)

# PDF 생성 함수
def create_meeting_report(topic, todo, contents, current_date, main_topic, current_user):
    # PDF 파일 생성
    pdf = SimpleDocTemplate("meeting_report.pdf", pagesize=A4)
    elements = []

    # 스타일 설정
    styles = getSampleStyleSheet()
    
    # 기본 글꼴 변경
    styles['Normal'].fontName = 'NanumGothic'
    styles['Title'].fontName = 'NanumGothic'
    styles['Heading2'].fontName = 'NanumGothic'
    
        # 양쪽 정렬 스타일 만들기
    justified_style = styles['Normal'].clone('Justified')
    justified_style.alignment = TA_JUSTIFY

    # 제목 추가
    elements.append(Paragraph("<b>회의록</b>", styles['Title']))
    elements.append(Spacer(1, 12))

    # 회의일시 및 참석자 테이블 생성
    data = [
        ['회의일시', current_date],
        ['참석자', current_user],
    ]
    table = Table(data, colWidths=[100, 300])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.whitesmoke),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'NanumGothic'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))

   # 회의 주제 섹션
    topic = topic.replace('\n', '<br/>')
    
    topic_data = [
        [Paragraph(main_topic, paragraph_style)],
        ['회의주제', Paragraph(topic, paragraph_style)]
    ]
    
    table_sub = Table(topic_data, colWidths=[100, 300], rowHeights=[None, None])
    table_sub.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (0, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),
        ('SPAN', (0, 0), (-1, 0)),  
        ('FONTNAME', (0, 0), (-1, -1), 'NanumGothic'),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),   # 왼쪽 패딩
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),  # 오른쪽 패딩
        ('TOPPADDING', (0, 0), (-1, -1), 10),    # 위쪽 패딩
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10), # 아래쪽 패딩
    ]))
    
    elements.append(table_sub)
    elements.append(Spacer(1, 20))

    # 회의 내용 섹션
    contents = contents.replace('\n', '<br/>')
    
    content_data = [
        ['회의 내용', '내용'],
        ['', Paragraph(contents, paragraph_style)]
    ]

    # 테이블 생성
    table_content = Table(content_data, colWidths=[100, 300], rowHeights=[20, None])
    
    table_content.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (0, 0), 'CENTER'),  # 병합된 셀 가운데 정렬
        ('SPAN', (0, 0), (0, 1)),  # 첫 번째 열의 1행과 2행 병합
        ('FONTNAME', (0, 0), (-1, -1), 'NanumGothic'),
        ('WORD_WRAP', (0, 0), (-1, -1), 'WORD_WRAP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),   # 왼쪽 패딩
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),  # 오른쪽 패딩
        ('TOPPADDING', (0, 0), (-1, -1), 10),    # 위쪽 패딩
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10), # 아래쪽 패딩
    ]))

    elements.append(table_content)
    elements.append(Spacer(1, 20))

    # 해야 할 일 섹션
    todo = todo.replace('\n', '<br/>')
    doit_data = [
        ['해야 할 일', '내용'],
        ['', Paragraph(todo, paragraph_style)]
    ]

    # 테이블 생성
    table_doit = Table(doit_data, colWidths=[100, 300], rowHeights=[20, None])
    
    table_doit.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (0, 0), 'CENTER'),  # 병합된 셀 가운데 정렬
        ('SPAN', (0, 0), (0, 1)),  # 첫 번째 열의 1행과 2행 병합
        ('FONTNAME', (0, 0), (-1, -1), 'NanumGothic'),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),   # 왼쪽 패딩
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),  # 오른쪽 패딩
        ('TOPPADDING', (0, 0), (-1, -1), 10),    # 위쪽 패딩
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10), # 아래쪽 패딩
    ]))

    elements.append(table_doit)
    elements.append(Spacer(1, 20))

    # PDF 저장
    pdf.build(elements)
    print("PDF 생성 완료")
    
# wkhtmltopdf 경로 설정
# config = pdfkit.configuration(
#     wkhtmltopdf="C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe")

# pdf 다운로드
@app.route('/summarize_download', methods=['POST'])
def export_pdf():
    try:
        html_content = request.json['htmlContent']
        print("html >>  ", html_content, file=sys.stdout)
        print("type >> ", type(html_content))
        pdf_filename = 'meeting_summary.pdf'

        Myoptions = {
            'encoding': 'utf-8',  # 인코딩 설정
            'footer-font-size': 10,
            'footer-font-name': 'MALGUN.TTF',  # 사용할 폰트 설정
            'quiet': '',
        }

        pdf = pdfkit.from_string(html_content, pdf_filename, options=Myoptions)

        # PDF를 클라이언트에 전송
        return send_file(pdf_filename, as_attachment=True)
    except Exception as e:
        return str(e)

# rag_search
@app.route('/rag_search', methods=['POST'])
def rag_search():
  data = request.get_json()
  query = data.get('query')

  if not query:
    return jsonify({'error': 'Query is required'}), 400

  answer = process_query(query)

  return jsonify({'answer': answer})


# FAISS 벡터 저장소 초기화
embedding_function = HuggingFaceEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# 임시문서 생성
tmp_docs = [LangChainDocument(page_content="dummy doc")]

# FAISS 인덱스 초기화
vector_store = FAISS.from_documents(tmp_docs, embedding_function)

# Google Search API를 사용하여 검색 결과 가져오기
def google_search(sh_query, num_results=10):
    global vector_store

    print("* query : ", sh_query)

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': GOOGLE_API_KEY,
        'cx': SEARCH_ENGINE_ID,
        'q': sh_query,
        'num': num_results,
        'start' : 1
    }
    response = requests.get(url, params = params)
    results = response.json().get('items', [])

    if not results:
        print("검색 결과가 없습니다.")
        return []

    if len(results) == 10:
      params['start'] = 11
      response = requests.get(url, params = params)
      results.extend(response.json().get('items', []))

    docs = []
    for index, item in enumerate(results):
        title = item.get('title')
        snippet = item.get('snippet')
        link = item.get('link')
        page_content = f"{title}\n{snippet}\nURL: {link}"
        
        doc_id = f"{index + 1}"
        docs.append(LangChainDocument(page_content=page_content, metadata={'id': doc_id}))  # LangChainDocument 객체로 저장
    
    if docs:
      # 문서를 벡터로 변환하여 FAISS 저장소에 추가
      vector_store.add_documents(docs)
      # FAISS 저장소 저장 (필요한 경우)
      vector_store.save_local("faiss_index")
    else:
        print("검색 결과가 없습니다. FAISS에 추가할 문서가 없습니다.")

    return docs

# 검색 결과를 토대로 BM25로 키워드 기반 검색
def bm25_search(query, documents, k=5):
    if not documents:
        return []

    tokenized_corpus = [doc.page_content.split() for doc in documents]  # 문서에서 page_content 추출
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_n = np.argsort(bm25_scores)[-k:]
    keyword_docs = [documents[i] for i in reversed(top_n)]
    
    return keyword_docs

# 검색 기능 (BM25 + FAISS)
def hybrid_search(query, google_docs, k=5):
    # BM25로 키워드 기반 검색
    keyword_docs = bm25_search(query, google_docs, k)

    # FAISS로 벡터 기반 검색
    semantic_docs = vector_store.similarity_search(query, k=k)

    # NoneType 체크 및 빈 리스트로 초기화
    if keyword_docs is None:
        keyword_docs = []
    if semantic_docs is None:
        semantic_docs = []

    # 두 검색 결과 결합
    combined_docs = keyword_docs + semantic_docs
    
    # 중복제거
    #unique_docs = list({doc.metadata['id']: doc for doc in combined_docs}.values())
    seen_ids = set()  # 이미 본 문서 ID를 저장할 집합
    deleted_ids = []  # 삭제된 문서의 ID를 저장할 리스트
    unique_docs = []
    
    for doc in combined_docs:
        doc_id = doc.metadata.get('id')
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            unique_docs.append(doc)
        else:
            deleted_ids.append(doc_id)  # 중복된 ID 저장

    print("삭제된 문서 ID:", deleted_ids)  # 삭제된 ID 출력
    
    print("### keyword_docs : ", keyword_docs)
    print("### semantic_docs : ", semantic_docs)
    print("### unipue_docs: ", unique_docs)
    
    return unique_docs

# GPT API를 사용하여 답변 생성
def generate_answer(docs, query):
    context = "\n".join([doc.page_content for doc in docs])

    print("**** context : ", context)
    
    prompt = f"""
    ### instruction ###
    질문에 대한 답변을 하되, 다음 조건을 반드시 지켜주세요:

    1. 제일 중요합니다. 우선적으로 컨텍스트를 사용하여 당신이 생각한 답변과 교차검증하여 꼭 사실관계를 확인하고 확인되지 않은 정보는 답변하지 마세요.
    2. 만약 관련깊은 컨텍스트가 없다면 그렇다고 해주세요.
    3. 신뢰할 수 있고 실존하는 url을 함께 알려주세요.
    4. 예를 들어, '베토벤에 대한 보고서 초고를 써줘' 처럼 단순 정보 전달이 아닐 경우, 컨텍스트를 활용하여 답변을 해주세요.


    컨텍스트:
    {context}

    질문: {query}
    답변:
    """
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "당신은 사용자가 입력한 질문에 대해 제공된 컨텍스트를 바탕으로 정확하고 상세한 답변을 제공하는 믿음직한 어시스턴트입니다.\
                                            사용자의 질문에 맞는 답변을 주세요."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content.strip()
    return answer

# 키워드
def searchKeyword(query):
    prompt = f"""
    ### instruction ###
    주어진 질문을 해결하기 위한 구글 검색에 적합한 키워드를 3~5개 생성하세요.
    질문에서 핵심 정보(인물, 작품명, 주제 등)를 추출하고, 검색에 불필요한 맥락 설명은 제거하여
    구글 검색 엔진이 효과적으로 검색할 수 있도록 최적화된 키워드로 만들어주세요.
    
    질문: {query}
    답변:
    """
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content.strip()
    return answer

# 전체 프로세스
def process_query(query):
    keywords = searchKeyword(query)
    
    print("*** keywords : ", keywords)
    
    google_docs = google_search(keywords)

    if not google_docs:
        dummy_content = "This is a dummy document"
        google_docs = [LangChainDocument(page_content=dummy_content, metadata={'id': 'dummy'})]
        vector_store = FAISS.from_documents(google_docs, embedding_function)
        generate_answer(google_docs, query)

    vector_store = FAISS.from_documents(google_docs, embedding_function)

    unique_docs = hybrid_search(keywords, google_docs, k=5)

    answer = generate_answer(unique_docs, query)

    return answer

if __name__ == '__main__':
    app.run(debug=True, port=8000)